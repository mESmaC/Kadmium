# AGENTS.md

Instructions for coding agents working in the Kadmium (`Cd-48`) repository. Read this file before touching code, docs, or configuration. Human contributors should read [CONTRIBUTING.md](CONTRIBUTING.md); this file adds the rules that agents get wrong most often. Tool-specific instruction files (`CLAUDE.md`, `.cursorrules`, Copilot instructions) must defer to this one rather than restate it.

Kadmium is a pure-Rust compute fabric that aggregates the heterogeneous machines a person already owns, such as a gaming PC, an M4 Mac mini, laptops, SBCs and a NAS, into one reactor for inference, training and reinforcement learning. The repository contains a working control plane, TCP worker transport, planners, reference workloads, a Control Room web workspace, a native desktop Companion, and Condenser, the inference workspace that routes prompts to local Ollama and hosted providers. Phase labels in the README describe code milestones, not release readiness.

---

## 1. The rule that outranks every other rule: never degrade model performance or quality

Kadmium exists so that people can run frontier models on hardware they own without accepting a worse model than the one they downloaded. Every default in this repository therefore resolves to the maximum the model and backend actually support. An agent that lowers a limit, guesses a cap, truncates input, prunes history, quantizes weights, or swaps in a smaller model to make a test pass or a request succeed has broken the product, even if every check is green.

### 1.1 What "maximum" means here

| Dimension | Required default | Where it is enforced today |
| :--- | :--- | :--- |
| Context / input window | The model's native maximum, read from backend metadata on every request. Ollama receives `num_ctx` equal to the model's declared context length. | `crates/kadmium-condenser/src/provider/ollama.rs` (`native_context`) |
| Output tokens | The provider's advertised maximum for that model. Ollama receives `num_predict: -1` (generate until stop). | `crates/kadmium-condenser/src/provider/mod.rs` (`maximum_output`, `require_output_maximum`) |
| Prompt truncation and context shifting | Disabled. Ollama receives `truncate: false` and `shift: false`. The full conversation is sent to every provider. | `ollama.rs` request body; `reject_context_override` for providers without a context option |
| Conversation history | Preserved in full. MCL4 history compaction and old tool-result pruning are off and require the explicit `compact_history` opt-in. | `crates/kadmium-condenser/src/types.rs` (`SamplingParams`), `provider/mcl4_coe.rs` |
| Generation timeout | None (`provider.timeout_secs: 0`). Catalog discovery stays bounded separately. | `provider/mod.rs` (`with_timeout`) |
| Weight precision | Lossless. Checkpoint normalization copies tensors into `.ktb` banks without casting; loading into Ollama never requests quantization and fails if the backend changes a dtype, name or shape. | `crates/kadmium-data/src/checkpoint.rs`, `crates/kadmium-condenser/src/local_models.rs` |
| Model and backend selection | Explicit. An allocation, precision or memory failure produces an error. No smaller model, shorter context, different backend or different quantization is chosen on the user's behalf. | `local_models.rs`, `docs/guides/local-checkpoints.md` |
| Sampling temperature | The provider default. Higher or lower temperature is not a quality lever and must not be changed as one. | `SamplingParams` |
| Kernel arithmetic | Bit-identical to the RepOps reference math. A native kernel changes speed, never bytes. | `crates/kadmium-cuda/README.md`, `docs/ARCHITECTURE.md` section 9.6 and 9.17 |

The regression tests for this contract live in `crates/kadmium-condenser/tests/quality_defaults.rs`. The test named `missing_maximum_never_submits_a_small_default` is the one agents are most tempted to weaken. Do not.

### 1.2 Forbidden changes

Do not do any of the following, in code, defaults, config, fixtures, UI copy or docs, unless the user explicitly asks for it in the task and the tradeoff is surfaced to the operator as described in 1.3:

- Introduce a numeric default for context, input, output, history length, or timeout. A missing provider maximum is an error that asks the operator for the model's documented limit. It is never a substituted `1024`, `4096`, `8192` or "reasonable" number.
- Enable truncation, sliding-window shifting, prompt compression, summarization of history, tool-result pruning, or any "compaction" by default.
- Clamp a requested value down silently. If a request exceeds the model's maximum, return an error naming both numbers.
- Quantize, cast, round, prune, or drop layers, experts or tensors during import, normalization, transfer, caching, or load. Expert caches, prefetch, delta links and coalesced reads must stay bit-identical to the uncached path.
- Fall back to a smaller model, a lower-precision variant, a shorter context, a different backend, or a "draft" model when the requested one cannot be loaded. Fail with the reason.
- Add a "fast mode", "lite mode", "economy" preset or "safe default" that trades quality for speed, memory or cost without the operator choosing it per setting.
- Advertise a maximum the executing backend cannot actually serve at full quality. A model whose window is only reachable with RoPE scaling, a reduced KV precision, or a swapped attention kernel does not have that window in Kadmium's sense. Report what is really served.
- Raise temperature, lower top-p, or change seeds to make a fixture or test pass.
- Reword UI or docs so a limit sounds like a feature. "Optimized context" is a lowered context.

If an upstream library, provider SDK or backend forces one of these behaviours, surface it as an error or a labelled limitation. Do not paper over it.

### 1.3 How an operator may lower a limit

Users may reduce limits themselves. Every control that can lower quality must satisfy all of the following:

1. **Blank means maximum.** The empty or unset state resolves to the model's maximum, not to a placeholder number. Follow the existing Condenser pattern: placeholders read "Full model context", "Maximum supported", "No limit".
2. **One setting, one effect.** Lowering output tokens must not also lower context, enable pruning or change precision. No cascades.
3. **The tradeoff is stated next to the control.** Say what is lost in the model's terms: "Input beyond this window is rejected, not truncated", "Responses stop at this many tokens", "Earlier turns are dropped from the prompt". Copy must say *you* chose it. Do not call it a recommendation.
4. **The ceiling is validated.** A value above the model's maximum is an error that names the maximum. A value of zero is an error.
5. **Reversal is one action.** A "Reset to maximum" control, or equivalent, returns every quality setting to its unset state.
6. **The setting is scoped and persisted honestly.** Per-task settings stay with the task and its forks; queued prompts keep the settings captured at submission.
7. **The lowered value is visible wherever the result is shown.** A response produced under a lowered limit must not be indistinguishable from a full-quality one in logs, run records or the UI.

Run-level budgets (step counts, unit limits, credits) are separate, explicit controls and are not quality settings; keep them separate.

### 1.4 Checklist before opening a PR that touches inference, Condenser, providers, checkpoints, caches, planners or kernels

- [ ] Every new default resolves to maximum or to an error. Grep your diff for literal token counts and justify each one.
- [ ] No new truncation, shift, compaction, quantization, cast or fallback path is reachable without an explicit opt-in field.
- [ ] `cargo test -p kadmium-condenser --locked` passes, including `quality_defaults.rs`, unchanged. If you had to edit that file, explain why in the PR body.
- [ ] Any new kernel or cache path has a digest or bitwise comparison against the reference path in its tests.
- [ ] UI copy and docs for any new or changed limit state the tradeoff and the maximum.
- [ ] The PR body contains a "Model quality impact" section. "None" is an acceptable answer only when the diff proves it.

---

## 2. Honesty about numbers and capabilities

Kadmium's docs separate three kinds of statements, and agents must keep them separate:

- **Implemented reference behavior.** Code that runs today, often on small synthetic tensors or reference kernels.
- **Modelled or simulated output.** Figures derived from rated bandwidth, fixture prices, or scenario inputs. Examples: USB4 goodput, failover timings, cloud spend.
- **Target architecture.** Designs described in `docs/ARCHITECTURE.md` or the roadmap that have no execution path yet.

Rules:

- Label every performance figure **Measured**, **Derived** or **Modelled**, and record measured runs in `docs/BENCHMARKS_2026-10.md` with date, hardware, configuration and provenance. Never upgrade a label in a doc, README, UI badge or commit message.
- Gateway `/v1/chat/completions`, `/v1/messages` and `/v1/embeddings` return diagnostic text and reference vectors. Do not describe them as OpenAI or Anthropic compatible endpoints, and do not write tests that pretend they generate.
- Platform classification is not execution support. A silicon or OS profile in `kadmium-core::platform` does not mean a native backend exists for it. Current native execution is six CUDA kernels and one Intel NPU MatMul, each with documented caveats.
- Managed services, payouts, procurement, settlement, WAN relays and public registries are planned or simulated. Do not add UI, MCP tool descriptions, or docs that present them as live.
- Do not change a number in `docs/NUMBERS_2026-10.md`, `docs/business/assumptions.json` or the economics model without updating the generated outputs and running `python scripts/business_model.py --check`.

---

## 3. Repository map

Workspace members and their place in the dependency graph. Dependencies point downward only; a lower crate never imports a higher one.

| Crate | Role | Depends on |
| :--- | :--- | :--- |
| `kadmium-core` | Node roles, capability vectors, hardware and platform classification, shared types | nothing in-workspace |
| `kadmium-cuda` | Pure-Rust CUDA kernels compiled by cuda-oxide; host side builds everywhere, `device/` is excluded from the workspace | nothing in-workspace |
| `kadmium-npu` | Portable NPU kernels, Intel Level Zero backend | nothing in-workspace |
| `kadmium-grid` | The single execution substrate: lifecycle, units, schedules, RepOps and Verde verification | core |
| `kadmium-sharding` | Asymmetric shard planner, $q^*$ placement, responsibility shifts, dataflow and pool slicing | core, grid |
| `kadmium-network` | KDM2 framing, TCP workers, USB-C delta bridges, striping, discovery | core |
| `kadmium-data` | `.ktb` tensor banks, checkpoint normalization, FTS3-style transfer scheduling | core, sharding |
| `kadmium-rl` | GRPO references, Dream-RSI replay | core, sharding |
| `kadmium-runtime` | Training and inference reference runtimes, DiLoCo, DeMo, expert cache, kernel dispatch | core, cuda, data, grid, network, npu, sharding |
| `kadmium-condenser` | Inference workspace: providers, local checkpoints, agent harness, embedded UI | core, data |
| `kadmium-gateway` | Axum coordinator, `/v1` diagnostics, cluster API, MCP, Control Room, Condenser mount | everything above except cuda and npu directly |
| `kadmium-cli` | The `kadmium` binary | gateway and most libraries |
| `kadmium-desktop` | Native Companion (WebView2, WKWebView, optional WebKitGTK) | gateway |

Add a mechanism to the crate that owns it. Do not duplicate a planner, cache, budget or transport in a higher crate because it is convenient. Memory budgets, owner offers and transfer windows are meant to have one authority; see issue #60 before adding another.

Key documents:

- `docs/ARCHITECTURE.md` is the design reference. Section 9 covers the grid substrate, RepOps, EPLB, DSpark, KV tiers, DeepGEMM, DeepEP, Ollama kernels, CUDA, NPU and InferSSD.
- `docs/api-reference.md` and `docs/cli-reference.md` are generated from route handlers and clap definitions respectively. Update them in the same PR as the code.
- `docs/guides/condenser-workspace.md` and `docs/guides/local-checkpoints.md` describe the maximum-quality contract from the operator's side.
- `design-system/1-voice.md` is the copy standard for UI, docs and README prose.

Ports: the coordinator and Control Room default to `7848`; standalone Condenser listens on loopback `7856` and is also mounted at `/condenser/` in the gateway.

---

## 4. Build, lint and test

The toolchain is pinned in `rust-toolchain.toml` (Rust `1.99.0`, edition 2024). `rustup` selects it automatically. Always pass `--locked`; CI does.

Full host sequence, identical to CI:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

CI runs on Ubuntu, Windows and macOS. Run the matrix-specific extras when your change touches them:

| If you changed | Also run |
| :--- | :--- |
| `kadmium-desktop` or `kadmium-condenser` on Linux | `cargo clippy -p kadmium-desktop -p kadmium-condenser --features linux-native --all-targets --locked -- -D warnings` (needs `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`) |
| Condenser stream or task behaviour | `node --test crates/kadmium-condenser/tests/workflow.test.cjs` (Node 24) |
| Condenser checkpoint or distributed compute UI | `cargo build -p kadmium-condenser --locked` then `node --test crates/kadmium-condenser/tests/local_compute.test.cjs` (Playwright) |
| Control Room or Companion UI | `cd crates/kadmium-gateway/tests/ui && npm ci && npx playwright install --with-deps chromium && npm test` |
| Control Room navigation or request recovery | `cargo build -p kadmium-gateway --example control_room_fixture --locked` then `node --test crates/kadmium-gateway/tests/control_room.test.cjs` |
| Model setup from either interface | same fixture build, then `node --test crates/kadmium-gateway/tests/model_setup.test.cjs` |
| CLI subcommands or Condenser engines | the CI smoke commands: `cargo run --locked --bin kadmium -- probe`, `simulate`, `dream`, `slice --pool-size 100 --slices 5`, `dataflow --model deepseek-v3-671b-moe --batch-tokens 512`, `cloud-shield`; `cargo run --locked --bin kadmium-condenser -- dsv41-profile --json` |
| Installers | `sh scripts/test_install.sh` or `./scripts/test_installers.ps1 -Executable target/debug/kadmium.exe` |
| Business or economics inputs | `python scripts/test_business_model.py` and `python scripts/business_model.py --check` (Python 3.12) |
| `crates/kadmium-cuda/device` | The device harness builds with `cargo oxide` on Linux or WSL2 with CUDA 13 and its own pinned toolchain. See `crates/kadmium-cuda/README.md`. It is not covered by workspace tests. |
| `kadmium-npu` hardware path | `cargo test --locked -p kadmium-npu --test intel_hardware -- --ignored` on a machine with a supported Intel NPU |

On Windows set `KADMIUM_TEST_BROWSER=msedge` for the workspace browser suite. Browser traces land in `crates/kadmium-gateway/tests/ui/test-results` and screenshots in `target/checkpoint-smoke`.

Formatting discipline: `rustfmt` on a `lib.rs` or `main.rs` follows `mod` declarations and reformats every child module. Format only files you changed, and check an untouched file through stdin (`git show HEAD:path | rustfmt --check --edition 2024`) so your diff contains only your change.

---

## 5. Code conventions

- **Pure Rust.** No Python, Node or shell at runtime for the shipped binaries. Node and Python appear only in tests and planning scripts. Adding a dependency needs a reason in the PR body; prefer the workspace-pinned versions in the root `Cargo.toml`.
- **Errors, not panics, on every path a user or worker can reach.** Use `thiserror` enums and the crate's `Result`. `unwrap`, `expect` and indexing on external data belong in tests only. Fallible readers must reject overflowing offsets and oversized declared lengths before allocating or reading. The provider response-size guard in `provider/mod.rs` is the pattern to copy; issue #52 is the failure to avoid.
- **Bounds and cancellation.** Every loop over network input, every catalog fetch, every grid run and every browser-driven workflow has a bound and a cancel path. Background tasks spawned for a connection must stop when the connection is cancelled; issue #49 is the counterexample.
- **Loopback and authentication.** Embedded Condenser APIs require a loopback transport peer and same-origin requests. Workers authenticate sessions. Do not widen a listener or relax an origin check to make a test easier.
- **Determinism.** Reductions, digests and tie-breaking are specified. A kernel, cache or transport change must leave digests identical; add the digest assertion to the test.
- **Keep layering.** See section 3. No reverse dependencies, no `pub` re-export of a lower crate's internals to dodge the graph.
- **No emoji, no exclamation marks** in product, docs, logs, error strings or UI. The GitHub README's section emoji are the single exception.
- **Vocabulary.** Status words are used identically in badges, logs and docs: resident, training, prefetching, joining, yielded, throttled, offline, faulted. `kadmium` lowercase in mono for the binary and crates; Kadmium capitalised for the product. Methods keep their published capitalisation: DiLoCo, DeMo, DisTrO, SWARM, SkipPipe, RepOps, Verde, GRPO.
- **Tests accompany mechanisms.** A new planner heuristic, transport behaviour, cache, or provider path ships with a test that exercises its failure mode, not only its happy path.

Documentation changes follow the rules in CONTRIBUTING.md: distinguish implemented from modelled from target, update references from source, check relative links and anchors, and avoid unrelated rewrites.

---

## 6. Delivery workflow: branch, verify, PR, and stay with it until merged

This is required for every task that changes the repository, not only large ones.

1. **Start from current `main`.** Create a branch named for the change, such as `feat/<topic>`, `fix/<topic>` or `docs/<topic>`. Never commit directly to `main`. If you are working in a worktree, run everything from that worktree.
2. **Keep the change focused.** One concern per PR. Do not bundle formatting of untouched files, unrelated refactors or drive-by doc edits.
3. **Run the checks** from section 4 that your diff touches, plus the full host sequence. Record the exact commands and results.
4. **Commit** with a conventional prefix matching the log: `feat(scope):`, `fix(scope):`, `docs:`, `test:`, `ci:`. Scope is the crate or area. The subject is imperative and under 72 characters.
5. **Push to `origin`** and **open a pull request against `main`** with `gh pr create` or the equivalent. The PR body must contain:
   - **Summary.** What changed and why, in the voice of section 5.
   - **Model quality impact.** State explicitly that no default was lowered, no truncation, compaction, quantization or fallback was added, and no advertised maximum exceeds what the backend serves. If any limit was changed, quote the new operator-facing tradeoff text.
   - **Claims.** Any number or capability statement added to docs or UI, with its Measured, Derived or Modelled label.
   - **Validation.** The commands you ran and their outcomes, including skipped platform-specific checks and why.
   - Related issues, using `Closes #N` only when the issue is fully resolved.
6. **Monitor the PR until it is merged.** The PR will be reviewed and either merged or returned with requested changes. Your task is not finished when the PR opens. Keep watching it:
   - Read every new comment, review and CI result, including reviews that arrive after you have already responded to earlier ones. More than one reviewer may post from the same account, so treat each comment on its own merits.
   - For each requested change: address it, push a new commit on the same branch (do not force-push over reviewed history), run the affected checks again, and reply on the thread stating what you changed and the validation you ran.
   - If you disagree with a request, say why on the thread with evidence from the code or docs. Do not silently ignore it and do not comply by lowering quality.
   - If CI fails, fix the cause. Do not retry blindly, weaken a test, or mark it ignored.
   - If `main` moves and the PR conflicts, merge or rebase onto `main`, resolve conflicts preserving both sides' intent, and re-run checks.
   - Do not merge your own PR. Merging is the reviewer's decision. Stop monitoring only when the PR is merged or the reviewer closes it.
7. **After merge,** if the change introduced a follow-up that you could not finish, open an issue describing it with the same honesty rules as section 2.

Agents operating in a harness that provides PR linking or watching tools should use them so the session is notified of review activity; otherwise poll the PR at a sensible cadence until it is merged.

---

## 7. Things to never do

- Lower any model quality default, add a hidden cap, or add a fallback that changes the model, precision, context or history. See section 1.
- Present modelled, simulated or fixture output as measured, or a roadmap item as shipped.
- Commit to `main`, force-push over reviewed commits, or merge your own PR.
- Weaken, skip or delete a test to get green, especially `quality_defaults.rs`, checkpoint integrity tests, digest comparisons and bridge resume tests.
- Reformat files you did not otherwise change.
- Add `unwrap` or unchecked indexing on network, file or provider data.
- Widen a listener, relax same-origin or loopback checks, or log secrets.
- Add emoji, exclamation marks, marketing adjectives, or the reactor metaphor beyond the one tagline.
- Describe `/v1` gateway routes as real LLM generation, or Kadmium Cloud, payouts, procurement or settlement as live.
