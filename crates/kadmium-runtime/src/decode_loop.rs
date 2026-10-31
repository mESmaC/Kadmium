//! The decode loop: the steps of one or several sessions as graphs through one
//! [`GridRunner`], step `t + 1` planned and queued while step `t` runs.
//!
//! DeepInfer's asynchronous scheduling runs the CPU's schedule of step `N + 1` during the
//! GPU's forward of step `N`, so the accelerator never waits for the scheduler. Here the
//! workers are the accelerator and the coordinator is the scheduler: in the pipelined loop the
//! coordinator builds the next step's graph while the current one executes on the workers and
//! submits it behind the current sink, so the first unit of step `t + 1` leaves in the same
//! loop turn that validates the last unit of step `t`. The sequential loop is the old shape,
//! one graph at a time through its own queue, with the next step planned only after the
//! previous one has settled. The figure of merit is the inter-step gap: the next step's first
//! dispatch minus this step's last return, on the coordinator's clock, measured on whatever
//! kernels the workers run (the reference math today).
//!
//! Several sessions interleave in the same queue: their graphs are independent, so the
//! matchmaker hands the stages of different sessions to different nodes at the same time.
//! That is batching across sessions at the level of units, not the merging of their rows into
//! one tensor; each session keeps its own rows and its own bytes.

use crate::{
    DraftVerifyReport, GridRunConfig, GridRunReport, GridRunner, LocalExecutor, TokenBatchInt8,
    verify_batch,
};
use kadmium_core::{HardwareNode, ModelSpec, NodeId};
use kadmium_grid::{ContentRef, ReplicaCatalog, WorkGraph, WorkUnitId, WorkerProfile, fnv1a_64};
use kadmium_network::LiveWorkerRegistration;
use kadmium_sharding::{
    AcceptanceAccounting, DraftProfile, DraftVerifyParams, ExpertPlacementPlan, ModelledDrafter,
    SessionDraft, StepGraphBuilder, StepsPerSecondProfile, SweepMeasurement, VerificationPolicy,
    VerificationSchedule, VerificationScheduler,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A session that drafts: the modelled drafter's profile and the seed its rows and coins come
/// from. A speculative session decodes one row; the verify batch is that row and its drafts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSpeculation {
    pub profile: DraftProfile,
    pub seed: u64,
}

/// What the loop needs to size verification: DSpark's policy and the steps-per-second profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationSetup {
    pub policy: VerificationPolicy,
    pub profile: StepsPerSecondProfile,
}

/// How the loop runs.
#[derive(Debug, Clone, Default)]
pub struct DecodeLoopOptions {
    /// Plan and queue each session's next step while its current one runs.
    pub pipelined: bool,
    /// Size the speculative sessions' verification each step; without it they draft nothing.
    pub verification: Option<VerificationSetup>,
}

impl DecodeLoopOptions {
    pub fn pipelined(pipelined: bool) -> Self {
        Self {
            pipelined,
            verification: None,
        }
    }

    pub fn with_verification(mut self, setup: VerificationSetup) -> Self {
        self.verification = Some(setup);
        self
    }
}

/// One session's decode steps as the loop will run them.
#[derive(Debug, Clone)]
pub struct DecodeSessionPlan {
    pub session_id: String,
    pub model: ModelSpec,
    pub placement: ExpertPlacementPlan,
    pub nodes: Vec<HardwareNode>,
    pub rows: u32,
    pub steps: u32,
    /// The epoch of the first step; each step after it adds one.
    pub first_epoch: u64,
    /// Which copy serves each replicated expert at one row.
    pub serving: BTreeMap<(u32, u32), NodeId>,
    /// The first step's input: the prompt batch (its first row alone for a speculative session).
    pub seed: TokenBatchInt8,
    /// Set for a session that drafts.
    pub speculation: Option<SessionSpeculation>,
}

/// The verification lengths of every step, scheduled across the speculative sessions.
struct StepSchedules<'a> {
    plans: &'a [DecodeSessionPlan],
    setup: Option<&'a VerificationSetup>,
    by_step: BTreeMap<u32, VerificationSchedule>,
}

impl<'a> StepSchedules<'a> {
    fn new(plans: &'a [DecodeSessionPlan], setup: Option<&'a VerificationSetup>) -> Self {
        Self {
            plans,
            setup,
            by_step: BTreeMap::new(),
        }
    }

    fn speculative(&self) -> Vec<usize> {
        self.plans
            .iter()
            .enumerate()
            .filter(|(_, p)| p.speculation.is_some())
            .map(|(i, _)| i)
            .collect()
    }

    /// The drafter's confidences for session `index` at `step`, `gamma` positions.
    fn confidences(&self, index: usize, step: u32) -> Vec<f64> {
        let Some(setup) = self.setup else {
            return Vec::new();
        };
        let Some(spec) = self.plans[index].speculation.as_ref() else {
            return Vec::new();
        };
        ModelledDrafter::confidences(
            &spec.profile,
            spec.seed,
            step as u64,
            setup.policy.max_draft,
        )
    }

    /// The schedule for `step`: DSpark's greedy admission over every speculative session's
    /// confidences from `confidence_lag` steps earlier (the first steps use their own).
    fn schedule(&mut self, step: u32) -> Option<&VerificationSchedule> {
        let setup = self.setup?;
        let speculative = self.speculative();
        if speculative.is_empty() {
            return None;
        }
        if !self.by_step.contains_key(&step) {
            let source_step = step.saturating_sub(setup.policy.confidence_lag);
            let drafts: Vec<SessionDraft> = speculative
                .iter()
                .map(|&i| SessionDraft {
                    session_id: self.plans[i].session_id.clone(),
                    confidences: self.confidences(i, source_step),
                })
                .collect();
            let schedule = VerificationScheduler::schedule(&drafts, &setup.profile, &setup.policy);
            self.by_step.insert(step, schedule);
        }
        self.by_step.get(&step)
    }

    /// The draft length of session `index` at `step`.
    fn length(&mut self, index: usize, step: u32) -> u32 {
        if step >= self.plans[index].steps {
            return 0;
        }
        let position = self.speculative().iter().position(|&i| i == index);
        match (position, self.schedule(step)) {
            (Some(pos), Some(schedule)) => schedule.length_for(pos),
            _ => 0,
        }
    }

    /// The verify parameters of session `index` at `step`, when it drafts.
    fn params(&mut self, index: usize, step: u32) -> Option<DraftVerifyParams> {
        let spec = self.plans[index].speculation.clone()?;
        let draft_length = self.length(index, step);
        let next_draft_length = self.length(index, step + 1);
        let mut confidences = self.confidences(index, step);
        confidences.truncate(draft_length as usize);
        let plan = &self.plans[index];
        Some(DraftVerifyParams {
            model_id: plan.model.id.clone(),
            epoch: plan.first_epoch + step as u64,
            session_id: plan.session_id.clone(),
            layer: plan.model.num_layers.saturating_sub(1),
            draft_length,
            confidences,
            next_draft_length,
            step: step as u64,
            seed: spec.seed,
        })
    }

    fn schedules(&self) -> Vec<VerificationSchedule> {
        self.by_step.values().cloned().collect()
    }
}

impl DecodeSessionPlan {
    fn graph_for(
        &self,
        step: u32,
        input: &ContentRef,
        verify: Option<&DraftVerifyParams>,
    ) -> WorkGraph {
        match verify {
            Some(params) => StepGraphBuilder::speculative_step_from(
                &self.model,
                &self.placement,
                &self.nodes,
                self.first_epoch + step as u64,
                &self.session_id,
                &self.serving,
                input,
                params,
            ),
            None => StepGraphBuilder::decode_step_from(
                &self.model,
                &self.placement,
                &self.nodes,
                self.first_epoch + step as u64,
                &self.session_id,
                self.rows,
                &self.serving,
                input,
            ),
        }
    }

    /// The bytes the first step reads: the prompt batch, or for a speculative session its
    /// first row followed by the first step's draft rows.
    fn first_input(&self, verify: Option<&DraftVerifyParams>) -> Result<Vec<u8>, String> {
        match (verify, &self.speculation) {
            (Some(params), Some(spec)) => {
                let row = self
                    .seed
                    .row_tensors()
                    .and_then(|rows| rows.into_iter().next())
                    .unwrap_or_else(|| self.seed.tensor.clone());
                Ok(
                    verify_batch(&row, self.seed.dim, spec.seed, 0, params.draft_length)?
                        .to_le_bytes(),
                )
            }
            _ => Ok(self.seed.to_le_bytes()),
        }
    }

    /// The sink unit of a step graph and the activation it emits.
    fn sink_of(graph: &WorkGraph) -> Option<(WorkUnitId, ContentRef)> {
        let sinks = graph.sinks();
        let sink = sinks.last()?;
        Some((
            sink.id.clone(),
            StepGraphBuilder::output_ref(&graph.id, &sink.id),
        ))
    }
}

/// One step of one session as the loop ran it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeStepOutcome {
    pub graph_id: String,
    pub epoch: u64,
    /// First dispatch and last return of the step's units, on the run config's clock.
    pub dispatched_at_ms: f64,
    pub returned_at_ms: f64,
    pub wall_ms: f64,
    pub units_total: usize,
    pub units_validated: usize,
    pub success: bool,
    /// The sink's output bytes, the next step's input.
    #[serde(skip)]
    pub output: Vec<u8>,
    pub output_fnv1a: u64,
    pub final_digest_fnv1a: u64,
    /// Draft positions verified and accepted this step (zero for a plain step), and the
    /// scheduler's expectation; the acceptance is modelled.
    #[serde(default)]
    pub proposed: u32,
    #[serde(default)]
    pub accepted: u32,
    #[serde(default)]
    pub expected_accepted: f64,
}

/// One session through the loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeSessionReport {
    pub session_id: String,
    pub rows: u32,
    pub steps: Vec<DecodeStepOutcome>,
    /// For each pair of consecutive steps, the next step's first dispatch minus this step's
    /// last return, in ms; measured on the coordinator.
    pub inter_step_gap_ms: Vec<f64>,
    pub mean_gap_ms: f64,
    pub max_gap_ms: f64,
    /// First dispatch of the first step to last return of the last step.
    pub wall_ms: f64,
    /// Tokens the session produced: its rows per step for a plain session; for a speculative
    /// one the golden token plus the accepted prefix per step (the full-acceptance count is
    /// in `accounting`).
    pub tokens: u64,
    pub success: bool,
    /// Present for a speculative session.
    #[serde(default)]
    pub accounting: Option<AcceptanceAccounting>,
}

/// What the loop ran and how it went.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeLoopReport {
    pub pipelined: bool,
    pub sessions: Vec<DecodeSessionReport>,
    pub mean_gap_ms: f64,
    pub max_gap_ms: f64,
    /// Whether units of two sessions were in flight at the same time.
    pub interleaved: bool,
    pub wall_ms: f64,
    pub tokens: u64,
    pub steps: u64,
    pub success: bool,
    /// The verification schedule of every step, in step order, when any session drafted.
    #[serde(default)]
    pub schedules: Vec<VerificationSchedule>,
    /// The speculative sessions' accounting summed.
    #[serde(default)]
    pub accounting: Option<AcceptanceAccounting>,
    /// Everything the loop ran, as one report.
    pub report: GridRunReport,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

impl DecodeLoopReport {
    /// This loop as one point of a sweep: its step time, throughput and per-user speed, with
    /// the full-acceptance throughput beside the accepted one when the sessions drafted.
    pub fn sweep_measurement(&self, sampled_unix_ms: u64) -> SweepMeasurement {
        let step_walls: Vec<f64> = self
            .sessions
            .iter()
            .flat_map(|s| s.steps.iter().map(|st| st.wall_ms))
            .collect();
        let step_ms_mean = if step_walls.is_empty() {
            0.0
        } else {
            step_walls.iter().sum::<f64>() / step_walls.len() as f64
        };
        let per_sec = |tokens: u64| {
            if self.wall_ms > 0.0 {
                tokens as f64 / self.wall_ms * 1000.0
            } else {
                0.0
            }
        };
        // A user is one row of a session's batch: its tokens over the session's wall.
        let per_user = if self.sessions.is_empty() {
            0.0
        } else {
            self.sessions
                .iter()
                .map(|s| {
                    if s.wall_ms > 0.0 {
                        s.tokens as f64 / s.rows.max(1) as f64 / s.wall_ms * 1000.0
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / self.sessions.len() as f64
        };
        let round = |v: f64| (v * 100.0).round() / 100.0;
        SweepMeasurement {
            step_ms_mean: round3(step_ms_mean),
            steps_per_second: round(if step_ms_mean > 0.0 {
                1000.0 / step_ms_mean
            } else {
                0.0
            }),
            tokens_per_sec: round(per_sec(self.tokens)),
            per_user_tokens_per_sec: round(per_user),
            wall_ms: self.wall_ms,
            tokens: self.tokens,
            mean_inter_step_gap_ms: self.mean_gap_ms,
            tokens_per_sec_full_acceptance: self
                .accounting
                .as_ref()
                .map(|a| round(per_sec(a.tokens_full_acceptance))),
            acceptance_rate: self
                .accounting
                .as_ref()
                .map(|a| (a.acceptance_rate * 10_000.0).round() / 10_000.0),
            sampled_unix_ms,
        }
    }
}

/// The span of a graph's units: first dispatch, last return.
fn graph_span(report: &GridRunReport, graph_id: &str) -> Option<(f64, f64)> {
    let mut first = f64::INFINITY;
    let mut last = f64::NEG_INFINITY;
    for t in report
        .traces
        .iter()
        .filter(|t| t.unit_id.belongs_to(graph_id))
    {
        first = first.min(t.dispatched_at_ms);
        last = last.max(t.returned_at_ms);
    }
    (first.is_finite() && last.is_finite()).then_some((first, last))
}

fn step_outcome(
    runner: &GridRunner<'_>,
    plan: &DecodeSessionPlan,
    graph: &WorkGraph,
    step: u32,
) -> DecodeStepOutcome {
    let report = runner.report_for(graph);
    let (dispatched_at_ms, returned_at_ms) = graph_span(&report, &graph.id).unwrap_or((0.0, 0.0));
    let output = DecodeSessionPlan::sink_of(graph)
        .and_then(|(_, out)| runner.blob(&out.content_id).map(|b| b.to_vec()))
        .unwrap_or_default();
    let final_digest_fnv1a = report
        .traces
        .iter()
        .rev()
        .find(|t| t.ok)
        .map_or(0, |t| t.digest_fnv1a);
    // A speculative step ends in a verify unit whose payload carries the acceptance.
    let verify: Option<DraftVerifyReport> = DecodeSessionPlan::sink_of(graph)
        .and_then(|(sink, _)| runner.payload_of(&sink))
        .and_then(|bytes| serde_json::from_slice(bytes).ok());
    DecodeStepOutcome {
        graph_id: graph.id.clone(),
        epoch: plan.first_epoch + step as u64,
        dispatched_at_ms,
        returned_at_ms,
        wall_ms: round3(returned_at_ms - dispatched_at_ms),
        units_total: report.units_total,
        units_validated: report.units_validated,
        success: report.success,
        output_fnv1a: fnv1a_64(&output),
        output,
        final_digest_fnv1a,
        proposed: verify.as_ref().map_or(0, |v| v.proposed),
        accepted: verify.as_ref().map_or(0, |v| v.accepted),
        expected_accepted: verify.as_ref().map_or(0.0, |v| v.expected_accepted),
    }
}

fn session_report(plan: &DecodeSessionPlan, steps: Vec<DecodeStepOutcome>) -> DecodeSessionReport {
    let gaps: Vec<f64> = steps
        .windows(2)
        .map(|w| round3(w[1].dispatched_at_ms - w[0].returned_at_ms))
        .collect();
    let mean = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f64>() / gaps.len() as f64
    };
    let max = gaps.iter().copied().fold(0.0, f64::max);
    let wall = match (steps.first(), steps.last()) {
        (Some(a), Some(b)) => round3(b.returned_at_ms - a.dispatched_at_ms),
        _ => 0.0,
    };
    let accounting = plan.speculation.as_ref().map(|_| {
        let mut acc = AcceptanceAccounting::default();
        for s in &steps {
            acc.record_step(s.proposed, s.accepted, s.expected_accepted);
        }
        acc
    });
    let tokens = match &accounting {
        Some(acc) => acc.tokens_accepted,
        None => steps.len() as u64 * plan.rows as u64,
    };
    DecodeSessionReport {
        session_id: plan.session_id.clone(),
        rows: plan.rows,
        success: !steps.is_empty() && steps.iter().all(|s| s.success),
        tokens,
        inter_step_gap_ms: gaps,
        mean_gap_ms: round3(mean),
        max_gap_ms: round3(max),
        wall_ms: wall,
        steps,
        accounting,
    }
}

/// The speculative sessions' accounting summed, when any session drafted.
fn total_accounting(sessions: &[DecodeSessionReport]) -> Option<AcceptanceAccounting> {
    let parts: Vec<&AcceptanceAccounting> = sessions
        .iter()
        .filter_map(|s| s.accounting.as_ref())
        .collect();
    if parts.is_empty() {
        return None;
    }
    let mut total = AcceptanceAccounting::default();
    for p in parts {
        total.steps += p.steps;
        total.proposed += p.proposed;
        total.accepted += p.accepted;
        total.expected_accepted += p.expected_accepted;
        total.tokens_full_acceptance += p.tokens_full_acceptance;
        total.tokens_accepted += p.tokens_accepted;
    }
    total.acceptance_rate = if total.proposed > 0 {
        total.accepted as f64 / total.proposed as f64
    } else {
        0.0
    };
    total.modelled = true;
    Some(total)
}

/// Whether the traces of two different sessions overlap in time.
fn interleaved(report: &GridRunReport, sessions: &[DecodeSessionPlan]) -> bool {
    let session_of = |id: &WorkUnitId| {
        sessions
            .iter()
            .position(|p| id.0.starts_with(&format!("{}/", p.session_id)))
    };
    let spans: Vec<(usize, f64, f64)> = report
        .traces
        .iter()
        .filter(|t| t.ok)
        .filter_map(|t| session_of(&t.unit_id).map(|s| (s, t.dispatched_at_ms, t.returned_at_ms)))
        .collect();
    spans.iter().any(|(sa, a0, a1)| {
        spans
            .iter()
            .any(|(sb, b0, b1)| sa != sb && a0 < b1 && b0 < a1)
    })
}

/// A session's step `t + 1` input: the previous step's sink output, seeded into the next runner
/// under the content id the new graph reads.
fn chained_input(plan: &DecodeSessionPlan, step: u32, previous: Option<&WorkGraph>) -> ContentRef {
    match previous.and_then(DecodeSessionPlan::sink_of) {
        Some((_, out)) => out,
        None => StepGraphBuilder::prompt_ref(&format!(
            "{}/step-{}",
            plan.session_id,
            plan.first_epoch + step as u64
        )),
    }
}

/// Runs the sessions' steps. Sequential: one graph at a time through its own queue, the next
/// step planned after the previous settles, sessions one after another. Pipelined: one runner
/// for everything, each session's next step planned and queued behind its current one while
/// that one runs, and the sessions' graphs interleaved. With `options.verification`, the
/// speculative sessions' draft lengths are scheduled per step by DSpark's rule.
pub async fn run_decode_loop(
    plans: &[DecodeSessionPlan],
    workers: &mut [WorkerProfile],
    catalog: &ReplicaCatalog,
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
    local: Option<&LocalExecutor>,
    config: &GridRunConfig,
    options: &DecodeLoopOptions,
) -> DecodeLoopReport {
    let pipelined = options.pipelined;
    let mut schedules = StepSchedules::new(plans, options.verification.as_ref());
    let mut sessions: Vec<DecodeSessionReport> = Vec::new();
    let combined_id = format!(
        "decode-loop/{}",
        plans
            .iter()
            .map(|p| p.session_id.as_str())
            .collect::<Vec<_>>()
            .join("+")
    );

    let report: GridRunReport = if !pipelined {
        let mut all_traces = Vec::new();
        let mut merged: Option<GridRunReport> = None;
        for (index, plan) in plans.iter().enumerate() {
            let mut outcomes = Vec::new();
            let mut previous: Option<WorkGraph> = None;
            let mut carried: Option<Vec<u8>> = None;
            for step in 0..plan.steps {
                let input = chained_input(plan, step, previous.as_ref());
                let verify = schedules.params(index, step);
                let graph = plan.graph_for(step, &input, verify.as_ref());
                let seed = match carried.take() {
                    Some(bytes) => bytes,
                    None => match plan.first_input(verify.as_ref()) {
                        Ok(bytes) => bytes,
                        Err(_) => break,
                    },
                };
                let mut runner = GridRunner::new(workers, catalog, live_workers, local, config);
                if runner.submit(&graph, &[], vec![(input, seed)]).is_err() {
                    break;
                }
                runner.run_until_settled().await;
                let outcome = step_outcome(&runner, plan, &graph, step);
                let step_report = runner.report_for(&graph);
                all_traces.extend(step_report.traces.iter().cloned());
                merged = Some(match merged.take() {
                    None => step_report,
                    Some(mut m) => {
                        m.units_total += step_report.units_total;
                        m.units_validated += step_report.units_validated;
                        m.units_failed += step_report.units_failed;
                        m.success = m.success && step_report.success;
                        m.traces.extend(step_report.traces);
                        m.events.extend(step_report.events);
                        m.disputes.extend(step_report.disputes);
                        m.compute_samples += step_report.compute_samples;
                        m.accesses.extend(step_report.accesses);
                        m.outputs.extend(step_report.outputs);
                        m.credits.extend(step_report.credits);
                        m.receipts.extend(step_report.receipts);
                        m
                    }
                });
                carried = Some(outcome.output.clone());
                let ok = outcome.success;
                outcomes.push(outcome);
                previous = Some(graph);
                if !ok {
                    break;
                }
            }
            sessions.push(session_report(plan, outcomes));
        }
        let mut m = merged.unwrap_or_else(|| GridRunReport {
            graph_id: combined_id.clone(),
            units_total: 0,
            units_validated: 0,
            units_failed: 0,
            success: false,
            wall_ms: 0.0,
            traces: Vec::new(),
            events: Vec::new(),
            disputes: Vec::new(),
            compute_samples: 0,
            accesses: Vec::new(),
            outputs: Vec::new(),
            credits: Vec::new(),
            receipts: Vec::new(),
        });
        m.graph_id = combined_id.clone();
        if let Some((first, last)) = m.traces.iter().fold(None, |acc: Option<(f64, f64)>, t| {
            Some(match acc {
                None => (t.dispatched_at_ms, t.returned_at_ms),
                Some((a, b)) => (a.min(t.dispatched_at_ms), b.max(t.returned_at_ms)),
            })
        }) {
            m.wall_ms = round3(last - first);
        }
        m
    } else {
        let mut runner = GridRunner::new(workers, catalog, live_workers, local, config);
        // Per session: the graphs submitted so far and the outcomes collected.
        let mut submitted: Vec<Vec<WorkGraph>> = vec![Vec::new(); plans.len()];
        let mut outcomes: Vec<Vec<DecodeStepOutcome>> = vec![Vec::new(); plans.len()];
        let mut settled: Vec<usize> = vec![0; plans.len()];
        let mut stuck: Vec<bool> = vec![false; plans.len()];
        for (i, plan) in plans.iter().enumerate() {
            let input = chained_input(plan, 0, None);
            let verify = schedules.params(i, 0);
            let graph = plan.graph_for(0, &input, verify.as_ref());
            let seed = match plan.first_input(verify.as_ref()) {
                Ok(bytes) => bytes,
                Err(_) => {
                    stuck[i] = true;
                    continue;
                }
            };
            if runner.submit(&graph, &[], vec![(input, seed)]).is_ok() {
                submitted[i].push(graph);
            } else {
                stuck[i] = true;
            }
        }
        loop {
            runner.dispatch_ready();
            // Keep one step planned and queued ahead of the one running, per session: the
            // planning overlaps the workers' execution of the current step.
            for (i, plan) in plans.iter().enumerate() {
                if stuck[i] {
                    continue;
                }
                let have = submitted[i].len();
                if have < plan.steps as usize && have < settled[i] + 2 {
                    let previous = submitted[i].last().cloned();
                    let input = chained_input(plan, have as u32, previous.as_ref());
                    let verify = schedules.params(i, have as u32);
                    let graph = plan.graph_for(have as u32, &input, verify.as_ref());
                    let external: Vec<(WorkUnitId, WorkUnitId)> = previous
                        .as_ref()
                        .and_then(DecodeSessionPlan::sink_of)
                        .and_then(|(sink, _)| graph.ready().first().map(|u| (u.id.clone(), sink)))
                        .into_iter()
                        .collect();
                    if runner.submit(&graph, &external, Vec::new()).is_ok() {
                        submitted[i].push(graph);
                    } else {
                        stuck[i] = true;
                    }
                }
            }
            // Collect the steps that settled.
            for (i, plan) in plans.iter().enumerate() {
                while settled[i] < submitted[i].len()
                    && runner.graph_settled(&submitted[i][settled[i]])
                {
                    let graph = &submitted[i][settled[i]];
                    let outcome = step_outcome(&runner, plan, graph, settled[i] as u32);
                    if !outcome.success {
                        stuck[i] = true;
                    }
                    outcomes[i].push(outcome);
                    settled[i] += 1;
                }
            }
            let all_done = plans
                .iter()
                .enumerate()
                .all(|(i, p)| stuck[i] || settled[i] >= p.steps as usize);
            if all_done {
                break;
            }
            if runner.in_flight() == 0 {
                if runner.dispatch_ready() == 0 {
                    // Nothing runs and nothing can be assigned: a session is blocked for good.
                    break;
                }
                continue;
            }
            if !runner.settle_one().await {
                break;
            }
        }
        for (i, plan) in plans.iter().enumerate() {
            sessions.push(session_report(plan, std::mem::take(&mut outcomes[i])));
        }
        runner.report_all(&combined_id)
    };

    let gaps: Vec<f64> = sessions
        .iter()
        .flat_map(|s| s.inter_step_gap_ms.iter().copied())
        .collect();
    let mean_gap_ms = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f64>() / gaps.len() as f64
    };
    DecodeLoopReport {
        pipelined,
        interleaved: interleaved(&report, plans),
        mean_gap_ms: round3(mean_gap_ms),
        max_gap_ms: round3(gaps.iter().copied().fold(0.0, f64::max)),
        wall_ms: report.wall_ms,
        tokens: sessions.iter().map(|s| s.tokens).sum(),
        steps: sessions.iter().map(|s| s.steps.len() as u64).sum(),
        success: !sessions.is_empty() && sessions.iter().all(|s| s.success),
        schedules: schedules.schedules(),
        accounting: total_accounting(&sessions),
        sessions,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DEFAULT_INITIAL_KTB_MERKLE_ROOT, GridDispatcher, LiveNodeWorkerServer, prompt_seed_hidden,
    };
    use kadmium_core::sample_personal_cluster;
    use kadmium_sharding::{ExecutionStrategy, ExpertPlacementPlanner};

    fn registration(node: &HardwareNode, addr: &str, rtt_ms: f64) -> LiveWorkerRegistration {
        LiveWorkerRegistration {
            node: node.clone(),
            worker_tcp_addr: addr.to_string(),
            worker_lan_ip: "127.0.0.1".into(),
            worker_tcp_port: addr
                .rsplit(':')
                .next()
                .and_then(|p| p.parse().ok())
                .unwrap_or(0),
            os_name: "test".into(),
            cpu_arch: "x86_64".into(),
            board_or_model: "test".into(),
            network_channel: "LocalLAN".into(),
            installed_version: "v0.1.0-cd48".into(),
            ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
            heartbeat_count: 1,
            registered_at_unix_ms: 1000,
            last_heartbeat_unix_ms: 1000,
            last_measured_rtt_ms: rtt_ms,
            executed_forward_stages: 0,
            executed_training_steps: 0,
            online: true,
        }
    }

    fn toy_moe() -> ModelSpec {
        ModelSpec {
            id: "toy-moe".into(),
            display_name: "Toy MoE".into(),
            total_params_billions: 4.0,
            num_layers: 4,
            is_moe: true,
            dense_layer_size_gb: 0.05,
            experts_per_layer: 8,
            active_experts_per_token: 2,
            single_expert_size_gb: 0.01,
            kv_cache_per_layer_gb: 0.01,
            training_activation_per_layer_gb: 0.02,
            lora_state_per_layer_gb: 0.01,
        }
    }

    fn plans(
        strategy: ExecutionStrategy,
        nodes: &[HardwareNode],
        sessions: usize,
        steps: u32,
        rows: u32,
    ) -> Vec<DecodeSessionPlan> {
        let model = toy_moe();
        let placement = ExpertPlacementPlanner::plan(&model, nodes, strategy, None);
        (0..sessions)
            .map(|i| {
                let values: Vec<f32> = (0..rows)
                    .flat_map(|r| prompt_seed_hidden(&format!("session {i} row {r}"), 64))
                    .collect();
                DecodeSessionPlan {
                    session_id: format!("sess-{i}"),
                    model: model.clone(),
                    placement: placement.clone(),
                    nodes: nodes.to_vec(),
                    rows,
                    steps,
                    first_epoch: 1,
                    serving: BTreeMap::new(),
                    seed: TokenBatchInt8::quantize(&values, rows, 64, 16).unwrap(),
                    speculation: None,
                }
            })
            .collect()
    }

    /// The loop in both modes over two loopback TCP workers, for both shapes and two sessions:
    /// the bytes of every step agree between modes and with the one-graph dispatcher, the
    /// pipelined loop interleaves the sessions, and its inter-step gap is written beside the
    /// sequential one for the benchmark ledger (measured on the reference kernels).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipelined_steps_match_sequential_bytes_and_interleave_sessions() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let mut ledger_rows = Vec::new();

        for strategy in [
            ExecutionStrategy::PipelineChain,
            ExecutionStrategy::ExpertParallel,
        ] {
            let plans = plans(strategy, &nodes, 2, 6, 2);
            let catalog = ReplicaCatalog::assume_placed(&plans[0].placement.rules, 1);
            let config = GridRunConfig::default();

            // Warm the channels and the kernels once, so neither mode pays for the first
            // connection; then collect three pairs for the benchmark ledger. Shared CI
            // scheduling and competing sessions can make either mode faster, so elapsed
            // time is diagnostic data, not a correctness assertion.
            let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
            let _ = run_decode_loop(
                &plans[..1],
                &mut workers,
                &catalog,
                &live_workers,
                None,
                &config,
                &DecodeLoopOptions::pipelined(false),
            )
            .await;
            let mut pairs = Vec::new();
            for _ in 0..3 {
                let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
                let sequential = run_decode_loop(
                    &plans,
                    &mut workers,
                    &catalog,
                    &live_workers,
                    None,
                    &config,
                    &DecodeLoopOptions::pipelined(false),
                )
                .await;
                assert!(sequential.success, "{:?}", sequential.report.traces);
                assert!(!sequential.pipelined);
                let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
                let pipelined = run_decode_loop(
                    &plans,
                    &mut workers,
                    &catalog,
                    &live_workers,
                    None,
                    &config,
                    &DecodeLoopOptions::pipelined(true),
                )
                .await;
                assert!(pipelined.success, "{:?}", pipelined.report.traces);
                assert!(pipelined.pipelined);
                assert!(pipelined.interleaved, "two sessions share the queue");
                assert!(
                    !sequential.interleaved,
                    "one graph at a time never overlaps"
                );
                pairs.push((sequential, pipelined));
            }
            let best = |f: &dyn Fn(&(DecodeLoopReport, DecodeLoopReport)) -> f64| {
                pairs.iter().map(f).fold(f64::INFINITY, f64::min)
            };
            let best_sequential = best(&|p| p.0.mean_gap_ms);
            let best_pipelined = best(&|p| p.1.mean_gap_ms);
            for (sequential, pipelined) in &pairs {
                for (s, p) in sequential.sessions.iter().zip(&pipelined.sessions) {
                    assert_eq!(s.session_id, p.session_id);
                    assert_eq!(s.steps.len(), 6);
                    assert_eq!(p.steps.len(), 6);
                    assert_eq!(s.inter_step_gap_ms.len(), 5);
                    assert_eq!(p.inter_step_gap_ms.len(), 5);
                    for session in [s, p] {
                        assert!(
                            session
                                .steps
                                .windows(2)
                                .all(|steps| steps[1].dispatched_at_ms >= steps[0].returned_at_ms),
                            "next step must wait for the preceding step's output"
                        );
                        assert!(session.inter_step_gap_ms.iter().all(|gap| gap.is_finite()));
                    }
                    for (a, b) in s.steps.iter().zip(&p.steps) {
                        assert_eq!(a.graph_id, b.graph_id);
                        assert_eq!(
                            a.output, b.output,
                            "{strategy:?} {}: pipelining changed the bytes of {}",
                            s.session_id, a.graph_id
                        );
                        assert_ne!(a.output_fnv1a, 0);
                    }
                    // Successive steps differ: the loop fed each step the previous output.
                    assert_ne!(s.steps[0].output, s.steps[1].output);
                    assert!(
                        p.inter_step_gap_ms.iter().all(|g| *g >= 0.0),
                        "{:?}",
                        p.inter_step_gap_ms
                    );
                }
                // The two sessions' outputs differ from each other: no row crossed sessions.
                assert_ne!(
                    pipelined.sessions[0].steps[5].output,
                    pipelined.sessions[1].steps[5].output
                );
            }
            let sequential = &pairs[0].0;
            // The one-graph dispatcher agrees with step 0 of session 0.
            let step0 = plans[0].graph_for(0, &chained_input(&plans[0], 0, None), None);
            let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
            let single = GridDispatcher::run_graph(
                &step0,
                &mut workers,
                &catalog,
                &live_workers,
                None,
                &GridRunConfig::default().with_seed(
                    StepGraphBuilder::prompt_ref(&step0.id),
                    plans[0].seed.to_le_bytes(),
                ),
            )
            .await;
            let (_, out) = DecodeSessionPlan::sink_of(&step0).unwrap();
            assert_eq!(
                single.output(&out.content_id).unwrap(),
                sequential.sessions[0].steps[0].output.as_slice()
            );

            for (sequential, pipelined) in &pairs {
                eprintln!(
                    "{strategy:?}: inter-step gap mean {:.3} ms (max {:.3}) sequential, {:.3} ms (max {:.3}) pipelined; wall {:.1} ms sequential, {:.1} ms pipelined",
                    sequential.mean_gap_ms,
                    sequential.max_gap_ms,
                    pipelined.mean_gap_ms,
                    pipelined.max_gap_ms,
                    sequential.wall_ms,
                    pipelined.wall_ms
                );
                ledger_rows.push(serde_json::json!({
                    "strategy": format!("{strategy:?}"),
                    "sessions": 2, "steps": 6, "rows": 2, "layers": 4, "experts_per_layer": 8,
                    "workers": "two loopback TCP workers (node-main-pc, node-m4-mac-mini profiles)",
                    "kernels": "reference-synthetic",
                    "sequential": { "mean_gap_ms": sequential.mean_gap_ms, "max_gap_ms": sequential.max_gap_ms, "wall_ms": sequential.wall_ms, "gaps": sequential.sessions.iter().map(|s| s.inter_step_gap_ms.clone()).collect::<Vec<_>>() },
                    "pipelined": { "mean_gap_ms": pipelined.mean_gap_ms, "max_gap_ms": pipelined.max_gap_ms, "wall_ms": pipelined.wall_ms, "interleaved": pipelined.interleaved, "gaps": pipelined.sessions.iter().map(|s| s.inter_step_gap_ms.clone()).collect::<Vec<_>>() },
                }));
            }
            eprintln!(
                "{strategy:?}: best measured gap {best_pipelined:.3} ms pipelined, {best_sequential:.3} ms sequential (informational)"
            );
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("kadmium-state")
            .join("benchmarks");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            dir.join("inter_step_gap.json"),
            serde_json::to_string_pretty(&ledger_rows).unwrap(),
        );
    }

    /// The sweep harness over two loopback TCP workers: batch widths 1, 2, 4 and 8 on one node
    /// and on two, each point a pipelined loop, written as a Pareto table of throughput
    /// against per-user speed with every row's configuration, for the benchmark ledger
    /// (measured on the reference kernels).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_over_batch_widths_and_node_counts_writes_a_pareto_table() {
        use kadmium_sharding::{
            CALIBRATION_KERNELS_REFERENCE, SweepConfiguration, SweepPoint, SweepTable,
        };
        let cluster = sample_personal_cluster();
        let both = [cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, both[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, both[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            both[0].id.0.clone(),
            registration(&both[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            both[1].id.0.clone(),
            registration(&both[1], &w1.local_addr.to_string(), 0.2),
        );
        let mut table = SweepTable::new("loopback-chain");
        let config = GridRunConfig::default();
        for node_count in [1usize, 2] {
            let nodes: Vec<HardwareNode> = both[..node_count].to_vec();
            let live: BTreeMap<String, LiveWorkerRegistration> = live_workers
                .iter()
                .filter(|(id, _)| nodes.iter().any(|n| n.id.0 == **id))
                .map(|(id, r)| (id.clone(), r.clone()))
                .collect();
            // Warm the channels once per node set.
            let warm = plans(ExecutionStrategy::PipelineChain, &nodes, 1, 2, 1);
            let catalog = ReplicaCatalog::assume_placed(&warm[0].placement.rules, 1);
            let mut workers = GridDispatcher::workers_from_registrations(&live);
            let _ = run_decode_loop(
                &warm,
                &mut workers,
                &catalog,
                &live,
                None,
                &config,
                &DecodeLoopOptions::pipelined(true),
            )
            .await;
            for rows in [1u32, 2, 4, 8] {
                let plans = plans(ExecutionStrategy::PipelineChain, &nodes, 1, 4, rows);
                let catalog = ReplicaCatalog::assume_placed(&plans[0].placement.rules, 1);
                let mut workers = GridDispatcher::workers_from_registrations(&live);
                let report = run_decode_loop(
                    &plans,
                    &mut workers,
                    &catalog,
                    &live,
                    None,
                    &config,
                    &DecodeLoopOptions::pipelined(true),
                )
                .await;
                assert!(report.success, "{:?}", report.report.traces);
                let mut node_ids: Vec<String> = nodes.iter().map(|n| n.id.0.clone()).collect();
                node_ids.sort();
                table.push(SweepPoint {
                    configuration: SweepConfiguration {
                        model_id: plans[0].model.id.clone(),
                        strategy: ExecutionStrategy::PipelineChain,
                        rows,
                        sessions: 1,
                        steps: 4,
                        node_count,
                        node_ids,
                        placement_digest: plans[0].placement.digest(),
                        versions: live
                            .iter()
                            .map(|(id, r)| (id.clone(), r.installed_version.clone()))
                            .collect(),
                        kernels: CALIBRATION_KERNELS_REFERENCE.to_string(),
                        pipelined: true,
                        speculative: None,
                        repository_state:
                            "claude/deepinfer-scheduling-7c30d3 over e0ad4df, uncommitted".into(),
                    },
                    measured: report.sweep_measurement(1),
                });
            }
        }
        assert_eq!(table.len(), 8);
        let curves = table.curves();
        assert_eq!(curves.len(), 2, "one curve per node set");
        assert!(curves.iter().all(|(_, idx)| idx.len() == 4));
        let pareto = table.pareto();
        assert!(
            pareto.iter().filter(|p| **p).count() >= 2,
            "each curve keeps at least one point"
        );
        for p in &table.points {
            assert!(p.measured.tokens_per_sec > 0.0);
            assert!(p.measured.per_user_tokens_per_sec > 0.0);
            assert_eq!(p.measured.tokens, 4 * p.configuration.rows as u64);
            assert!(p.measured.tokens_per_sec_full_acceptance.is_none());
        }
        let md = table.to_markdown("Loopback sweep");
        assert!(md.contains("reference-synthetic"));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("kadmium-state")
            .join("benchmarks");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("sweep_loopback.md"), &md);
        let _ = std::fs::write(dir.join("sweep_loopback.jsonl"), table.to_jsonl());
        eprintln!("{md}");
    }

    /// Speculative sessions on the reference math over two TCP workers. A drafter that is never
    /// right verifies nothing and decodes exactly as a plain session does; a drafter that is
    /// often right verifies the batches the scheduler sized, and every step's next input is the
    /// plain decode of the verified row it accepted, which is batch invariance applied to a
    /// verify batch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn speculative_sessions_verify_what_the_scheduler_admits_and_keep_the_bytes() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let plain_plans = plans(ExecutionStrategy::PipelineChain, &nodes, 1, 4, 1);
        let catalog = ReplicaCatalog::assume_placed(&plain_plans[0].placement.rules, 1);
        let config = GridRunConfig::default();
        // A flat profile: every position with positive survival pays, so the scheduler admits
        // the whole draft of a drafter that is often right.
        let setup = VerificationSetup {
            policy: VerificationPolicy {
                max_draft: 3,
                ..VerificationPolicy::default()
            },
            profile: StepsPerSecondProfile::new(vec![(1, 100.0), (8, 100.0)], "modelled", "test"),
        };

        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let plain = run_decode_loop(
            &plain_plans,
            &mut workers,
            &catalog,
            &live_workers,
            None,
            &config,
            &DecodeLoopOptions::pipelined(true),
        )
        .await;
        assert!(plain.success);
        assert!(plain.accounting.is_none());
        assert!(plain.schedules.is_empty());

        // Never right: nothing verified, the same bytes as the plain session.
        let mut never = plain_plans.clone();
        never[0].speculation = Some(SessionSpeculation {
            profile: DraftProfile::none(),
            seed: 11,
        });
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let none = run_decode_loop(
            &never,
            &mut workers,
            &catalog,
            &live_workers,
            None,
            &config,
            &DecodeLoopOptions::pipelined(true).with_verification(setup.clone()),
        )
        .await;
        assert!(none.success, "{:?}", none.report.traces);
        let acc = none
            .accounting
            .as_ref()
            .expect("a speculative session accounts");
        assert_eq!((acc.proposed, acc.accepted), (0, 0));
        assert_eq!(acc.tokens_full_acceptance, acc.tokens_accepted);
        assert_eq!(acc.tokens_accepted, 4);
        assert!(acc.modelled);
        assert!(
            none.schedules
                .iter()
                .all(|s| s.batch_tokens == 1 && s.pairs_considered == 0)
        );
        for (a, b) in plain.sessions[0].steps.iter().zip(&none.sessions[0].steps) {
            assert_eq!(
                a.output, b.output,
                "{}: a drafter that drafts nothing changes nothing",
                a.graph_id
            );
            assert_eq!(b.proposed, 0);
            assert_eq!(b.units_total, a.units_total + 1, "one verify unit per step");
        }

        // Often right: the scheduler admits the whole draft; each step verifies four rows and
        // the next input is the plain decode of the accepted row.
        let mut often = plain_plans.clone();
        let seed = 11u64;
        often[0].speculation = Some(SessionSpeculation {
            profile: DraftProfile::code(),
            seed,
        });
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let spec = run_decode_loop(
            &often,
            &mut workers,
            &catalog,
            &live_workers,
            None,
            &config,
            &DecodeLoopOptions::pipelined(true).with_verification(setup.clone()),
        )
        .await;
        assert!(spec.success, "{:?}", spec.report.traces);
        assert_eq!(spec.schedules.len(), 4);
        assert!(
            spec.schedules
                .iter()
                .all(|s| s.lengths == vec![3] && s.batch_tokens == 4)
        );
        let acc = spec.accounting.as_ref().unwrap();
        assert_eq!(acc.proposed, 12);
        assert!(
            acc.accepted > 0,
            "a drafter right 85% of the time has some prefix accepted"
        );
        assert!(acc.accepted <= acc.proposed);
        assert_eq!(acc.tokens_full_acceptance, 16);
        assert_eq!(acc.tokens_accepted, 4 + acc.accepted);
        assert_eq!(spec.tokens, acc.tokens_accepted);
        assert!(acc.tau() > 1.0 && acc.tau() <= 4.0);

        let steps = &spec.sessions[0].steps;
        let mut input_row = often[0].seed.tensor.clone();
        for (t, step) in steps.iter().enumerate() {
            assert_eq!(step.proposed, 3, "step {t}");
            let verified = verify_batch(&input_row, 64, seed, t as u64, 3).unwrap();
            let rows = verified.row_tensors().unwrap();
            // The plain decode of the row the step accepted.
            let chosen = TokenBatchInt8 {
                rows: 1,
                dim: 64,
                tensor: rows[step.accepted as usize].clone(),
            };
            let graph = StepGraphBuilder::decode_step(
                &often[0].model,
                &often[0].placement,
                &nodes,
                900 + t as u64,
                "oracle",
                1,
            );
            let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
            let oracle = GridDispatcher::run_graph(
                &graph,
                &mut workers,
                &catalog,
                &live_workers,
                None,
                &GridRunConfig::default().with_seed(
                    StepGraphBuilder::prompt_ref(&graph.id),
                    chosen.to_le_bytes(),
                ),
            )
            .await;
            assert!(oracle.success);
            let (_, out) = DecodeSessionPlan::sink_of(&graph).unwrap();
            let expected =
                TokenBatchInt8::from_le_bytes(oracle.output(&out.content_id).unwrap()).unwrap();
            let next = TokenBatchInt8::from_le_bytes(&step.output).unwrap();
            let next_len = if t + 1 < steps.len() { 3 } else { 0 };
            assert_eq!(
                next.rows,
                1 + next_len,
                "step {t} emits the next verify batch"
            );
            let next_rows = next.row_tensors().unwrap();
            assert_eq!(
                next_rows[0].to_le_bytes(),
                expected.tensor.to_le_bytes(),
                "step {t}: the next input is the plain decode of the row accepted at position {}",
                step.accepted
            );
            input_row = next_rows[0].clone();
        }
    }
}
