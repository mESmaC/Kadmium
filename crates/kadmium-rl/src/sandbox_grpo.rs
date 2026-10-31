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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_error: Option<String>,
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
    const MAX_CANDIDATE_CODE_BYTES: usize = 4096;
    const MAX_CANDIDATE_TOKENS: usize = 512;
    const MAX_COMPILED_OPS: usize = 512;
    const MAX_PARSE_DEPTH: usize = 32;

    /// Compiles a candidate polynomial/arithmetic expression over `x` (or explicit `SandboxOp`
    /// sequence) into deterministic `SandboxOp` instructions, or returns a compile/validation error.
    pub fn compile_candidate_code(code: &str) -> Result<Vec<SandboxOp>, String> {
        let trimmed = code.trim();
        if trimmed.is_empty() {
            return Err("Empty candidate code snippet".into());
        }
        if trimmed.len() > Self::MAX_CANDIDATE_CODE_BYTES {
            return Err(format!(
                "Candidate code length ({} bytes) exceeds sandbox limit ({} bytes)",
                trimmed.len(),
                Self::MAX_CANDIDATE_CODE_BYTES
            ));
        }

        let forbidden_keywords = [
            "panic!",
            "todo!",
            "unimplemented!",
            "unreachable!",
            "unsafe",
            "loop",
            "while",
            "std::process",
            "std::fs",
            "std::net",
        ];
        for kw in forbidden_keywords {
            if trimmed.contains(kw) {
                return Err(format!(
                    "Forbidden construct '{kw}' in deterministic WasiMicroSandbox candidate"
                ));
            }
        }

        if let Some(ops) = Self::try_parse_stack_dsl(trimmed)? {
            if ops.len() > Self::MAX_COMPILED_OPS {
                return Err(format!(
                    "Compiled SandboxOp count ({}) exceeds limit ({})",
                    ops.len(),
                    Self::MAX_COMPILED_OPS
                ));
            }
            return Ok(ops);
        }

        let expr_body = Self::extract_expression_body(trimmed)?;
        let tokens = Self::tokenize_arith_expr(expr_body)?;
        if tokens.is_empty() {
            return Err("Candidate expression produced no executable tokens".into());
        }
        if tokens.len() > Self::MAX_CANDIDATE_TOKENS {
            return Err(format!(
                "Candidate expression token count ({}) exceeds sandbox limit ({})",
                tokens.len(),
                Self::MAX_CANDIDATE_TOKENS
            ));
        }
        let mut cursor = 0usize;
        let mut ops = Vec::new();
        Self::parse_add_sub(&tokens, &mut cursor, &mut ops, 0)?;
        if cursor != tokens.len() {
            return Err(format!(
                "Unexpected trailing token '{}' at position {}",
                tokens[cursor], cursor
            ));
        }
        if ops.len() > Self::MAX_COMPILED_OPS {
            return Err(format!(
                "Compiled SandboxOp count ({}) exceeds limit ({})",
                ops.len(),
                Self::MAX_COMPILED_OPS
            ));
        }
        Ok(ops)
    }

    fn try_parse_stack_dsl(trimmed: &str) -> Result<Option<Vec<SandboxOp>>, String> {
        let inner = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(trimmed);
        let first_ident: String = inner
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
            .collect();
        let looks_like_dsl = matches!(
            first_ident.as_str(),
            "PushInputX"
                | "PushConst"
                | "Add"
                | "Sub"
                | "Mul"
                | "Dup"
                | "SandboxOp::PushInputX"
                | "SandboxOp::PushConst"
                | "SandboxOp::Add"
                | "SandboxOp::Sub"
                | "SandboxOp::Mul"
                | "SandboxOp::Dup"
        );
        if !looks_like_dsl {
            return Ok(None);
        }

        let mut ops = Vec::new();
        for raw_part in inner.split([',', ';', '\n']) {
            let part = raw_part.trim().trim_start_matches("SandboxOp::").trim();
            if part.is_empty() {
                continue;
            }
            match part {
                "PushInputX" => ops.push(SandboxOp::PushInputX),
                "Add" => ops.push(SandboxOp::Add),
                "Sub" => ops.push(SandboxOp::Sub),
                "Mul" => ops.push(SandboxOp::Mul),
                "Dup" => ops.push(SandboxOp::Dup),
                _ if part.starts_with("PushConst") => {
                    let arg = part
                        .strip_prefix("PushConst")
                        .unwrap_or("")
                        .trim()
                        .strip_prefix('(')
                        .and_then(|s| s.strip_suffix(')'))
                        .ok_or_else(|| format!("Malformed PushConst instruction: '{part}'"))?;
                    let val: i64 = arg
                        .trim()
                        .parse()
                        .map_err(|_| format!("Invalid PushConst integer literal: '{arg}'"))?;
                    ops.push(SandboxOp::PushConst(val));
                }
                other => {
                    return Err(format!("Unknown SandboxOp instruction '{other}'"));
                }
            }
            if ops.len() > Self::MAX_COMPILED_OPS {
                return Err(format!(
                    "Compiled SandboxOp count ({}) exceeds limit ({})",
                    ops.len(),
                    Self::MAX_COMPILED_OPS
                ));
            }
        }

        if ops.is_empty() {
            return Err("Empty SandboxOp program".into());
        }
        Ok(Some(ops))
    }

    fn extract_expression_body(trimmed: &str) -> Result<&str, String> {
        let mut s = trimmed;
        if let Some(open_brace) = s.find('{') {
            let close_brace = s
                .rfind('}')
                .ok_or_else(|| "Unclosed '{' in candidate function body".to_string())?;
            if close_brace <= open_brace {
                return Err("Malformed brace block in candidate code".into());
            }
            let header = &s[..open_brace];
            if !header.trim().starts_with("fn ") && !header.trim().starts_with("pub fn ") {
                return Err(format!(
                    "Unsupported block header '{}' in candidate code",
                    header.trim()
                ));
            }
            s = s[open_brace + 1..close_brace].trim();
        } else if let Some(rest) = s.strip_prefix("|x|") {
            s = rest.trim();
        } else if let Some(rest) = s.strip_prefix("|x: i64|") {
            s = rest.trim();
        }

        s = s.trim_end_matches(';').trim();
        if let Some(rest) = s.strip_prefix("return ") {
            s = rest.trim();
        }
        if s.contains(';') {
            return Err(
                "Multi-statement imperative blocks are not supported; provide a pure expression over x or SandboxOp list"
                    .into(),
            );
        }
        Ok(s)
    }

    fn tokenize_arith_expr(expr: &str) -> Result<Vec<String>, String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = expr.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if tokens.len() >= Self::MAX_CANDIDATE_TOKENS {
                return Err(format!(
                    "Candidate expression token count exceeds sandbox limit ({})",
                    Self::MAX_CANDIDATE_TOKENS
                ));
            }
            match c {
                '+' | '-' | '(' | ')' => {
                    tokens.push(c.to_string());
                    i += 1;
                }
                '*' => {
                    if i + 1 < chars.len() && chars[i + 1] == '*' {
                        tokens.push("^".to_string());
                        i += 2;
                    } else {
                        tokens.push("*".to_string());
                        i += 1;
                    }
                }
                '^' => {
                    tokens.push("^".to_string());
                    i += 1;
                }
                '0'..='9' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '_') {
                        i += 1;
                    }
                    let num_str: String = chars[start..i]
                        .iter()
                        .copied()
                        .filter(|ch| *ch != '_')
                        .collect();
                    tokens.push(num_str);
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    let ident: String = chars[start..i].iter().collect();
                    if ident != "x" {
                        return Err(format!(
                            "Unsupported identifier '{ident}' in candidate expression (only variable 'x' is bound)"
                        ));
                    }
                    tokens.push(ident);
                }
                other => {
                    return Err(format!(
                        "Unsupported character '{other}' in candidate expression"
                    ));
                }
            }
        }
        Ok(tokens)
    }

    fn check_depth_and_ops(depth: usize, ops: &[SandboxOp]) -> Result<(), String> {
        if depth > Self::MAX_PARSE_DEPTH {
            return Err(format!(
                "Expression nesting depth ({depth}) exceeds sandbox limit ({})",
                Self::MAX_PARSE_DEPTH
            ));
        }
        if ops.len() > Self::MAX_COMPILED_OPS {
            return Err(format!(
                "Compiled SandboxOp count ({}) exceeds limit ({})",
                ops.len(),
                Self::MAX_COMPILED_OPS
            ));
        }
        Ok(())
    }

    fn parse_add_sub(
        tokens: &[String],
        cursor: &mut usize,
        ops: &mut Vec<SandboxOp>,
        depth: usize,
    ) -> Result<(), String> {
        Self::check_depth_and_ops(depth, ops)?;
        Self::parse_mul(tokens, cursor, ops, depth + 1)?;
        while *cursor < tokens.len() {
            match tokens[*cursor].as_str() {
                "+" => {
                    *cursor += 1;
                    Self::parse_mul(tokens, cursor, ops, depth + 1)?;
                    ops.push(SandboxOp::Add);
                    Self::check_depth_and_ops(depth, ops)?;
                }
                "-" => {
                    *cursor += 1;
                    Self::parse_mul(tokens, cursor, ops, depth + 1)?;
                    ops.push(SandboxOp::Sub);
                    Self::check_depth_and_ops(depth, ops)?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_mul(
        tokens: &[String],
        cursor: &mut usize,
        ops: &mut Vec<SandboxOp>,
        depth: usize,
    ) -> Result<(), String> {
        Self::check_depth_and_ops(depth, ops)?;
        Self::parse_unary(tokens, cursor, ops, depth + 1)?;
        while *cursor < tokens.len() && tokens[*cursor] == "*" {
            *cursor += 1;
            Self::parse_unary(tokens, cursor, ops, depth + 1)?;
            ops.push(SandboxOp::Mul);
            Self::check_depth_and_ops(depth, ops)?;
        }
        Ok(())
    }

    fn parse_unary(
        tokens: &[String],
        cursor: &mut usize,
        ops: &mut Vec<SandboxOp>,
        depth: usize,
    ) -> Result<(), String> {
        Self::check_depth_and_ops(depth, ops)?;
        if *cursor >= tokens.len() {
            return Err("Unexpected end of expression".into());
        }
        match tokens[*cursor].as_str() {
            "+" => {
                *cursor += 1;
                Self::parse_unary(tokens, cursor, ops, depth + 1)
            }
            "-" => {
                *cursor += 1;
                ops.push(SandboxOp::PushConst(0));
                Self::parse_unary(tokens, cursor, ops, depth + 1)?;
                ops.push(SandboxOp::Sub);
                Self::check_depth_and_ops(depth, ops)
            }
            _ => Self::parse_power(tokens, cursor, ops, depth + 1),
        }
    }

    fn parse_power(
        tokens: &[String],
        cursor: &mut usize,
        ops: &mut Vec<SandboxOp>,
        depth: usize,
    ) -> Result<(), String> {
        Self::check_depth_and_ops(depth, ops)?;
        Self::parse_primary(tokens, cursor, ops, depth + 1)?;
        if *cursor < tokens.len() && tokens[*cursor] == "^" {
            *cursor += 1;
            if *cursor >= tokens.len() {
                return Err("Missing exponent after '^'".into());
            }
            let exp: u32 = tokens[*cursor]
                .parse()
                .map_err(|_| format!("Invalid integer exponent '{}'", tokens[*cursor]))?;
            *cursor += 1;
            if exp == 0 {
                // x^0 == 1: multiply by 0 and add 1
                ops.push(SandboxOp::PushConst(0));
                ops.push(SandboxOp::Mul);
                ops.push(SandboxOp::PushConst(1));
                ops.push(SandboxOp::Add);
            } else if exp <= 8 {
                for _ in 1..exp {
                    ops.push(SandboxOp::Dup);
                }
                for _ in 1..exp {
                    ops.push(SandboxOp::Mul);
                }
            } else {
                return Err(format!("Exponent {exp} exceeds sandbox unroll limit (8)"));
            }
            Self::check_depth_and_ops(depth, ops)?;
        }
        Ok(())
    }

    fn parse_primary(
        tokens: &[String],
        cursor: &mut usize,
        ops: &mut Vec<SandboxOp>,
        depth: usize,
    ) -> Result<(), String> {
        Self::check_depth_and_ops(depth, ops)?;
        if *cursor >= tokens.len() {
            return Err("Unexpected end of expression".into());
        }
        let tok = &tokens[*cursor];
        if tok == "(" {
            *cursor += 1;
            Self::parse_add_sub(tokens, cursor, ops, depth + 1)?;
            if *cursor >= tokens.len() || tokens[*cursor] != ")" {
                return Err("Missing closing ')' in expression".into());
            }
            *cursor += 1;
            Ok(())
        } else if tok == "x" {
            *cursor += 1;
            ops.push(SandboxOp::PushInputX);
            Self::check_depth_and_ops(depth, ops)
        } else if let Ok(val) = tok.parse::<i64>() {
            *cursor += 1;
            ops.push(SandboxOp::PushConst(val));
            Self::check_depth_and_ops(depth, ops)
        } else {
            Err(format!("Unexpected token '{tok}' in expression"))
        }
    }

    /// Compiles and verifies a candidate code string against `test_cases`.
    /// Malformed or non-compilable code returns 0 passed assertions, `timed_out_or_trapped = false`,
    /// `verifiable_reward = 0.0`, and populates `compile_error`.
    pub fn verify_candidate_code(
        &self,
        rollout_id: impl Into<String>,
        host_node: impl Into<String>,
        candidate_code: &str,
        test_cases: &[(i64, i64)],
    ) -> SandboxVerificationResult {
        let rollout_id = rollout_id.into();
        let host_node = host_node.into();
        match Self::compile_candidate_code(candidate_code) {
            Ok(program) => {
                self.verify_candidate_program(rollout_id, host_node, &program, test_cases)
            }
            Err(err) => SandboxVerificationResult {
                rollout_id,
                host_node,
                passed_assertions: 0,
                total_assertions: test_cases.len().max(1),
                fuel_consumed: 0,
                timed_out_or_trapped: false,
                verifiable_reward: 0.0,
                compile_error: Some(err),
            },
        }
    }

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
            compile_error: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_candidate_code_rejects_deeply_nested_parens_and_unary_minus_without_stack_overflow()
    {
        let deep_parens = format!("{}x{}", "(".repeat(200), ")".repeat(200));
        let err_parens = WasiMicroSandbox::compile_candidate_code(&deep_parens).unwrap_err();
        assert!(
            err_parens.contains("exceeds sandbox limit"),
            "unexpected error for deep parens: {err_parens}"
        );

        let deep_unary = format!("{}x", "-".repeat(200));
        let err_unary = WasiMicroSandbox::compile_candidate_code(&deep_unary).unwrap_err();
        assert!(
            err_unary.contains("exceeds sandbox limit"),
            "unexpected error for deep unary minus: {err_unary}"
        );

        let oversized = "x + 1 ".repeat(1000);
        let err_oversized = WasiMicroSandbox::compile_candidate_code(&oversized).unwrap_err();
        assert!(err_oversized.contains("exceeds sandbox limit"));
    }
}
