//! Reproducible CPU training experiment, independent of the gateway's synthetic training path.
//!
//! Each island owns a complete two-shard model and persistent local AdamW moments. Only
//! parameters are replaced at the outer barrier; resetting moments here changes the algorithm.

use kadmium_runtime::{DiLoCoConfig, DiLoCoSession, IslandUpdate, ParameterShard};
use serde_json::{Value, json};

const SAMPLES_PER_ISLAND: usize = 32;
const INNER_LEARNING_RATE: f32 = 0.02;

struct Learner {
    parameters: [f32; 2],
    first_moment: [f32; 2],
    second_moment: [f32; 2],
    steps: u32,
    samples: Vec<(f32, f32)>,
}

impl Learner {
    fn new(island: u32, islands: u32) -> Self {
        // Each island sees a different part of the input range, with the same ground truth.
        let shift = (island as f32 - (islands - 1) as f32 / 2.0) / islands as f32;
        let samples = (0..SAMPLES_PER_ISLAND)
            .map(|index| {
                let x = -1.0 + 2.0 * index as f32 / (SAMPLES_PER_ISLAND - 1) as f32 + shift;
                (x, 2.0 * x + 1.0)
            })
            .collect();
        Self {
            parameters: [0.0; 2],
            first_moment: [0.0; 2],
            second_moment: [0.0; 2],
            steps: 0,
            samples,
        }
    }

    fn step(&mut self) {
        let mut gradient = [0.0; 2];
        for &(x, y) in &self.samples {
            let residual = self.parameters[0] * x + self.parameters[1] - y;
            gradient[0] += 2.0 * residual * x / self.samples.len() as f32;
            gradient[1] += 2.0 * residual / self.samples.len() as f32;
        }
        self.steps += 1;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        for (index, gradient) in gradient.into_iter().enumerate() {
            self.first_moment[index] = beta1 * self.first_moment[index] + (1.0 - beta1) * gradient;
            self.second_moment[index] =
                beta2 * self.second_moment[index] + (1.0 - beta2) * gradient * gradient;
            let mean = self.first_moment[index] / (1.0 - beta1.powf(self.steps as f32));
            let variance = self.second_moment[index] / (1.0 - beta2.powf(self.steps as f32));
            self.parameters[index] *= 1.0 - INNER_LEARNING_RATE * 0.01;
            self.parameters[index] -= INNER_LEARNING_RATE * mean / (variance.sqrt() + 1e-8);
        }
    }
}

fn shards(parameters: [f32; 2]) -> Vec<ParameterShard> {
    vec![
        ParameterShard {
            shard_id: "weight".into(),
            values: vec![parameters[0]],
        },
        ParameterShard {
            shard_id: "bias".into(),
            values: vec![parameters[1]],
        },
    ]
}

fn parameters(shards: &[ParameterShard]) -> Result<[f32; 2], String> {
    let value = |id: &str| {
        shards
            .iter()
            .find(|shard| shard.shard_id == id && shard.values.len() == 1)
            .map(|shard| shard.values[0])
            .ok_or_else(|| format!("snapshot is missing the scalar {id} shard"))
    };
    Ok([value("weight")?, value("bias")?])
}

fn loss(learners: &[Learner], parameters: [f32; 2]) -> f64 {
    let sum: f64 = learners
        .iter()
        .flat_map(|learner| &learner.samples)
        .map(|&(x, y)| {
            let residual = parameters[0] as f64 * x as f64 + parameters[1] as f64 - y as f64;
            residual * residual
        })
        .sum();
    sum / (learners.len() * SAMPLES_PER_ISLAND) as f64
}

fn experiment(inner_steps: u32, rounds: u32, islands: u32) -> Result<Value, String> {
    if !(1..=10_000).contains(&inner_steps) {
        return Err("--inner-steps must be between 1 and 10000".into());
    }
    if !(1..=1_000).contains(&rounds) {
        return Err("--rounds must be between 1 and 1000".into());
    }
    if !(2..=64).contains(&islands) {
        return Err("--islands must be between 2 and 64".into());
    }
    let total_local_steps = u64::from(inner_steps) * u64::from(rounds) * u64::from(islands);
    if total_local_steps > 2_000_000 {
        return Err("reference experiment is limited to 2000000 total local steps".into());
    }
    let config = DiLoCoConfig {
        inner_steps,
        outer_learning_rate: 0.7,
        outer_momentum: 0.9,
    };
    let island_ids: Vec<_> = (0..islands)
        .map(|index| format!("island-{index}"))
        .collect();
    let mut session = DiLoCoSession::new(config, island_ids.clone(), shards([0.0; 2]))
        .map_err(|error| error.to_string())?;
    let mut learners: Vec<_> = (0..islands)
        .map(|index| Learner::new(index, islands))
        .collect();
    let initial_loss = loss(&learners, [0.0; 2]);
    let mut loss_history = vec![initial_loss];
    for _ in 0..rounds {
        let snapshot = session.snapshot();
        let committed = parameters(&snapshot.shards)?;
        for (index, learner) in learners.iter_mut().enumerate() {
            learner.parameters = committed;
            for _ in 0..inner_steps {
                learner.step();
            }
            session
                .submit(IslandUpdate {
                    island_id: island_ids[index].clone(),
                    round: snapshot.round,
                    baseline_id: snapshot.baseline_id.clone(),
                    local_steps: inner_steps,
                    shards: shards(learner.parameters),
                })
                .map_err(|error| error.to_string())?;
        }
        loss_history.push(loss(&learners, parameters(&session.snapshot().shards)?));
    }
    let snapshot = session.snapshot();
    let status = session.status();
    let final_parameters = parameters(&snapshot.shards)?;
    let per_step_syncs = u64::from(inner_steps) * u64::from(rounds);
    let round_payload = status.parameter_payload_estimates.total_bytes_per_round;
    Ok(json!({
        "workload": "deterministic_cpu_linear_regression",
        "objective": "mean squared error for y = 2*x + 1; nonidentical input ranges per island",
        "config": status.config,
        "inner_optimizer": {"kind": "AdamW", "learning_rate": INNER_LEARNING_RATE,
            "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.01,
            "state_retained_across_rounds": true},
        "islands": islands,
        "samples_per_island": SAMPLES_PER_ISLAND,
        "round": snapshot.round,
        "baseline_id": snapshot.baseline_id,
        "parameter_shards": snapshot.shards,
        "initial_loss": initial_loss,
        "final_loss": loss(&learners, final_parameters),
        "loss_history": loss_history,
        "local_steps_per_island": learners.iter().map(|learner| learner.steps).collect::<Vec<_>>(),
        "total_local_steps": total_local_steps,
        "synchronization_count": status.synchronization_count,
        "per_step_synchronization_baseline": per_step_syncs,
        "synchronization_frequency_reduction": inner_steps,
        "parameter_payload_estimates": status.parameter_payload_estimates,
        "per_step_parameter_payload_baseline_bytes": round_payload * per_step_syncs,
        "limitations": "Single-process CPU reference; byte figures count FP32 upload/broadcast payloads only, not measured network traffic. No LLM, GPU, async RL, inference speedup, or end-to-end performance claim."
    }))
}

pub(crate) fn run(
    inner_steps: u32,
    rounds: u32,
    islands: u32,
    as_json: bool,
) -> Result<(), String> {
    let report = experiment(inner_steps, rounds, islands)?;
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
        );
    } else {
        println!(
            "DiLoCo CPU reference: {islands} islands, {rounds} rounds, {inner_steps} local AdamW steps per round"
        );
        println!("Objective: y = 2*x + 1, two parameter shards, nonidentical island data");
        println!(
            "Mean squared error: {:.6} -> {:.6}",
            report["initial_loss"].as_f64().unwrap(),
            report["final_loss"].as_f64().unwrap()
        );
        println!(
            "Outer synchronizations: {} (per-step baseline: {}; {}x fewer)",
            report["synchronization_count"],
            report["per_step_synchronization_baseline"],
            inner_steps
        );
        println!(
            "Estimated FP32 parameter payload: {} bytes (per-step baseline: {} bytes)",
            report["parameter_payload_estimates"]["completed_round_bytes"],
            report["per_step_parameter_payload_baseline_bytes"]
        );
        println!(
            "Local AdamW moments persist across rounds. Use --json for loss history and learned parameters."
        );
        println!(
            "Single-process CPU reference; payload estimates are not measured network traffic or LLM performance."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_islands_learn_with_fewer_synchronizations() {
        let report = experiment(20, 25, 3).unwrap();
        let initial = report["initial_loss"].as_f64().unwrap();
        let final_loss = report["final_loss"].as_f64().unwrap();
        assert!(
            final_loss < initial * 0.01,
            "loss did not converge: {report}"
        );
        assert_eq!(report["local_steps_per_island"], json!([500, 500, 500]));
        assert_eq!(report["synchronization_count"], 25);
        assert_eq!(report["per_step_synchronization_baseline"], 500);
        assert_eq!(
            report["parameter_payload_estimates"]["completed_round_bytes"],
            1_200
        );
        assert_eq!(report["per_step_parameter_payload_baseline_bytes"], 24_000);
        assert_eq!(report["parameter_shards"].as_array().unwrap().len(), 2);
        assert_ne!(Learner::new(0, 3).samples, Learner::new(1, 3).samples);
    }

    #[test]
    fn rejects_unbounded_or_empty_experiments() {
        assert!(experiment(0, 1, 2).is_err());
        assert!(experiment(1, 0, 2).is_err());
        assert!(experiment(1, 1, 1).is_err());
        assert!(experiment(10_000, 1_000, 64).is_err());
    }
}
