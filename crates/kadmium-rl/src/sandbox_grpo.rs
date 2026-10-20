//! Deterministic WASI Micro-Sandbox & GRPO Verified Rollout Executor (`kadmium-rl::sandbox_grpo`).

use crate::compute_grpo_group_advantages;
use serde::{Deserialize, Serialize};

/// Deterministic instruction set executed inside a fuel-metered WASI micro-sandbox
/// on `rk3588-sbc-bramble` or `legacy-thinkpad-t480` to score RL code/math rollouts safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxOp {
    /// Pushes the test-case input integer $x$ onto the sandbox stack.
    PushInputX,
    /// Pushes a literal constant onto the sandbox stack.
    PushConst(i64),
    /// Pops top two values and pushes $a + b$.
    Add,
    /// Pops top two values and pushes $a - b$.
    Sub,
    /// Pops top two values and pushes $a \times b$.
    Mul,
    /// Duplicates the top value on the stack (`x -> x, x`).
    Dup,
}

/// Outcome of executing a candidate RL rollout program inside the fuel-metered sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxVerificationResult {
    pub rollout_id: String,
    pub host_node: String,
    pub passed_assertions: usize,
    pub total_assertions: usize,
    pub fuel_consumed: u32,
    pub timed_out_or_trapped: bool,
    pub verifiable_reward: f64,
}

/// Deterministic, fuel-metered micro-sandbox for verifying RL code/math rollouts on edge CPUs/SBCs.
pub struct WasiMicroSandbox {
    pub max_fuel: u32,
    pub max_stack_depth: usize,
}

impl Default for WasiMicroSandbox {
    fn default() -> Self {
        Self {
            max_fuel: 256,
            max_stack_depth: 64,
        }
    }
}

impl WasiMicroSandbox {
    /// Executes `program` against `input_x` with strict instruction fuel and stack bounds.
    pub fn execute_metered(
        &self,
        program: &[SandboxOp],
        input_x: i64,
    ) -> Result<(i64, u32), String> {
        let mut stack: Vec<i64> = Vec::with_capacity(16);
        let mut fuel = 0u32;

        for &op in program {
            fuel += 1;
            if fuel > self.max_fuel {
                return Err("FuelLimitExceeded".into());
            }
            match op {
                SandboxOp::PushInputX => {
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(input_x);
                }
                SandboxOp::PushConst(c) => {
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(c);
                }
                SandboxOp::Dup => {
                    let top = *stack.last().ok_or("StackUnderflow")?;
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(top);
                }
                SandboxOp::Add => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_add(b));
                }
                SandboxOp::Sub => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_sub(b));
                }
                SandboxOp::Mul => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_mul(b));
                }
            }
        }

        let result = stack.pop().ok_or("EmptyReturnStack")?;
        Ok((result, fuel))
    }

    /// Scores a candidate rollout program against a suite of `(input_x, expected_y)` test cases.
    pub fn verify_candidate_program(
        &self,
        rollout_id: impl Into<String>,
        host_node: impl Into<String>,
        program: &[SandboxOp],
        test_cases: &[(i64, i64)],
    ) -> SandboxVerificationResult {
        let mut passed = 0usize;
        let mut total_fuel = 0u32;
        let mut trapped = false;

        for &(x, expected) in test_cases {
            match self.execute_metered(program, x) {
                Ok((actual, fuel)) => {
                    total_fuel += fuel;
                    if actual == expected {
                        passed += 1;
                    }
                }
                Err(_) => {
                    trapped = true;
                    break;
                }
            }
        }

        let total = test_cases.len().max(1);
        let base_reward = ((passed as f64) / (total as f64)) * 0.90;
        // Conciseness bonus (up to +0.10) when all assertions pass
        let efficiency_bonus = if passed == total && !trapped {
            (0.10 * (1.0 - (program.len() as f64 / 20.0).min(1.0))).max(0.0)
        } else {
            0.0
        };
        let reward = ((base_reward + efficiency_bonus).min(1.0) * 1000.0).round() / 1000.0;

        SandboxVerificationResult {
            rollout_id: rollout_id.into(),
            host_node: host_node.into(),
            passed_assertions: passed,
            total_assertions: total,
            fuel_consumed: total_fuel,
            timed_out_or_trapped: trapped,
            verifiable_reward: reward,
        }
    }
}

/// Complete report for a verified GRPO step ($G = 8$ rollouts) executed across disaggregated RL nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoVerifiedStepReport {
    pub task_description: String,
    pub group_size: usize,
    pub rollouts: Vec<SandboxVerificationResult>,
    pub group_advantages: Vec<f64>,
    pub best_rollout_id: String,
    pub mean_group_reward: f64,
}

/// Executes a complete Disaggregated GRPO verification and advantage calculation step.
pub struct GrpoEpochExecutor;

impl GrpoEpochExecutor {
    /// Runs a group of 8 synthesized candidate programs attempting to compute $f(x) = x^2 + 2x + 1$
    /// across `rk3588-sbc-bramble`, `legacy-thinkpad-t480`, and `home-nas-vault` WASI sandboxes.
    pub fn run_verified_grpo_step() -> GrpoVerifiedStepReport {
        let sandbox = WasiMicroSandbox::default();
        // Target polynomial: f(x) = (x + 1)^2 = x^2 + 2x + 1
        let test_cases = [(0, 1), (1, 4), (2, 9), (3, 16), (-1, 0), (5, 36)];

        let candidates: [(&str, &str, Vec<SandboxOp>); 8] = [
            // Rollout 1: Optimal factored form (x + 1) * (x + 1) -> 4 ops, 100% pass!
            (
                "rollout-g01",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                ],
            ),
            // Rollout 2: Expanded form x*x + 2*x + 1 -> 9 ops, 100% pass
            (
                "rollout-g02",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(2),
                    SandboxOp::Mul,
                    SandboxOp::Add,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 3: Partial solution x^2 + 1 (passes x=0 only)
            (
                "rollout-g03",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 4: Linear approximation 3x + 1 (passes x=0, x=1)
            (
                "rollout-g04",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(3),
                    SandboxOp::Mul,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 5: Malformed stack underflow trap
            (
                "rollout-g05",
                "home-nas-vault",
                vec![SandboxOp::PushInputX, SandboxOp::Add],
            ),
            // Rollout 6: Constant 1 (passes x=0)
            (
                "rollout-g06",
                "home-nas-vault",
                vec![SandboxOp::PushConst(1)],
            ),
            // Rollout 7: Exact factored form with redundant +0
            (
                "rollout-g07",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushConst(0),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 8: Off-by-one x^2 + 2x
            (
                "rollout-g08",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(2),
                    SandboxOp::Mul,
                    SandboxOp::Add,
                ],
            ),
        ];

        let mut rollouts = Vec::with_capacity(8);
        for (id, host, prog) in &candidates {
            rollouts.push(sandbox.verify_candidate_program(*id, *host, prog, &test_cases));
        }

        let rewards: Vec<f64> = rollouts.iter().map(|r| r.verifiable_reward).collect();
        let advantages = compute_grpo_group_advantages(&rewards);
        let mean_reward =
            ((rewards.iter().sum::<f64>() / (rewards.len() as f64)) * 1000.0).round() / 1000.0;

        let best_rollout_id = rollouts
            .iter()
            .max_by(|a, b| {
                a.verifiable_reward
                    .partial_cmp(&b.verifiable_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.rollout_id.clone())
            .unwrap_or_else(|| "rollout-g01".into());

        GrpoVerifiedStepReport {
            task_description: "Synthesize minimal bytecode for f(x) = x^2 + 2x + 1".into(),
            group_size: rollouts.len(),
            rollouts,
            group_advantages: advantages,
            best_rollout_id,
            mean_group_reward: mean_reward,
        }
    }
}
