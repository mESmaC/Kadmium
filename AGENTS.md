# AGENTS.md

Instructions for coding agents working in the Kadmium (`Cd-48`) repository. Read this file before touching code, docs, or configuration. Human contributors should read [CONTRIBUTING.md](CONTRIBUTING.md); this file adds the rules that agents get wrong most often. Tool-specific instruction files (`CLAUDE.md`, `.cursorrules`, Copilot instructions) must defer to this one rather than restate it.

Kadmium is a pure-Rust compute fabric that aggregates the heterogeneous machines a person already owns, such as a gaming PC, an M4 Mac mini, laptops, SBCs and a NAS, into one reactor for inference, training and reinforcement learning. The repository contains a working control plane, TCP worker transport, planners, reference workloads, a Control Room web workspace, a native desktop Companion, and Condenser, the inference workspace that routes prompts to local Ollama and hosted providers. Phase labels in the README describe code milestones, not release readiness.

---

## 1. The rule that outranks every other rule: never degrade model performance or quality

Kadmium exists so that people can run frontier models on hardware they own without accepting a worse model than the one they downloaded. Every default in this repository therefore resolves to the maximum the model and backend actually support. An agent that lowers a limit, guesses a cap, truncates input, prunes history, quantizes weights, or swaps in a smaller model to make a test pass or a request succeed has broken the product, even if every check is green.

### 1.1 What "maximum" means here

| Dimension | Required default | Enforcement anchor |
| :--- | :--- | :--- |
| Context / input window | The model's native maximum, read from backend metadata on every request. Ollama receives `num_ctx` equal to the model's declared context length. A request above the maximum is an error naming both numbers. | `crates/kadmium-condenser/src/provider/ollama.rs` (`native_context`) |
| Output tokens | The model's maximum. Where the wire protocol lets the request omit an output limit (OpenAI-compatible, MCL4), omit it when none is advertised; omission already means the server's maximum. Where the protocol requires a number (Anthropic `max_tokens`), send the advertised maximum, or fail with an error asking for the documented limit. Ollama receives `num_predict: -1` (generate until stop). | `crates/kadmium-condenser/src/provider/mod.rs` (`maximum_output`, `advertised_output_limit`, `require_output_maximum`) |
| Prompt truncation and context shifting | Disabled. Ollama receives `truncate: false` and `shift: false` as top-level `/api/chat` request fields, not `options` keys, where Ollama would ignore them. The full conversation is sent to every provider. | `ollama.rs` request body; `reject_context_override` for providers without a context option |
| Conversation history | Preserved in full. MCL4 history compaction and old tool-result pruning are off and require the explicit `compact_history` opt-in. | `crates/kadmium-condenser/src/types.rs` (`SamplingParams`), `provider/mcl4_coe.rs` |
| Generation timeout | No total generation timeout (`provider.timeout_secs: 0`). Connect and idle-read timeouts stay bounded so a hung provider cannot hold a run forever; those are transport safety, not quality settings. Catalog discovery stays bounded separately. | `provider/mod.rs` (`with_timeout`, client builder) |
| Weight precision | Lossless. Checkpoint normalization copies tensors into `.ktb` banks without casting; loading into Ollama never requests quantization and fails if the backend changes a dtype, name or shape. A source that was quantized before import is refused unless the operator accepts it for that import (`ImportOptions::accept_pre_quantized`); the manifest records it as `pre_quantized`, and every view labels it. Issue #61 tracks the remaining end-to-end gap: writing the checkpoint's native context length into the generated Modelfile (`PARAMETER num_ctx`). | `crates/kadmium-data/src/checkpoint.rs`, `crates/kadmium-condenser/src/local_models.rs` |
| Model and backend selection | Explicit. An allocation, precision or memory failure is an explicit error. It is never a CPU spill, a silent reload at a smaller context, a smaller model, a different backend or a different quantization chosen on the user's behalf. | `local_models.rs`, `docs/guides/local-checkpoints.md` |
| Sampling temperature | The provider default. Higher or lower temperature is not a quality lever and must not be changed as one. | `SamplingParams` |
| Kernel arithmetic | Bit-identical to the RepOps reference math. A native kernel changes speed, never bytes. | `crates/kadmium-cuda/README.md`, `docs/ARCHITECTURE.md` section 9.6 and 9.17 |

Anchor status: the Condenser maximum-quality defaults (`native_context`, `advertised_output_limit`, `require_output_maximum`, `reject_context_override`, `compact_history`, the `quality_defaults.rs` tests and `model_setup.test.cjs`) are delivered by PRs #54 and #64. Until those merge, `main` does not contain them; the contract above still applies, and the anchors name where to look once they land. Update this table when the anchors move.

The regression tests for this contract live in `crates/kadmium-condenser/tests/quality_defaults.rs`. The test named `missing_maximum_never_submits_a_small_default` is the one agents are most tempted to weaken. Do not.

### 1.2 Forbidden changes

Do not do any of the following, in code, defaults, config, fixtures, UI copy or docs, unless the user explicitly asks for it in the task and the tradeoff is surfaced to the operator as described in 1.3:

- Introduce a numeric default for context, input, output, history length, or timeout. Where the API lets you omit a limit, omit it; omission is the maximum. Where the API requires a value, use the advertised maximum, or fail with an error asking for the documented limit. Never substitute a guessed `1024`, `4096`, `8192` or "reasonable" number, and never reintroduce a hard error on APIs that accept omission; that error broke api.openai.com, DeepSeek, LM Studio and llama.cpp once already.
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
- [ ] `cargo test -p kadmium-condenser --locked` passes, including `quality_defaults.rs` once it is on `main`, unchanged. If you had to edit that file, explain why in the PR body.
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
- Label a statement backed by a Lean theorem in `formal/` **Proved (model)** or **Proved (Aeneas)**, according to how the theorem is tied to the Rust code (see `formal/README.md`). Do not write "proved", "verified" or "optimal" for anything without such a theorem listed in `formal/KadmiumFormal/Audit.lean`. An optimality proof holds under its stated cost model; it is not a performance figure and never stands in for a **Measured** benchmark.
- Gateway `/v1/chat/completions`, `/v1/messages` and `/v1/embeddings` return diagnostic text and reference vectors. Do not describe them as OpenAI or Anthropic compatible endpoints, and do not write tests that pretend they generate.
- Platform classification is not execution support. A silicon or OS profile in `kadmium-core::platform` does not mean a native backend exists for it. Current native execution is six CUDA kernels and one Intel NPU MatMul, each with documented caveats.
- Managed services, payouts, procurement, settlement, WAN relays and public registries are planned or simulated. Do not add UI, MCP tool descriptions, or docs that present them as live.
- Do not change a number in `docs/NUMBERS_2026-10.md`, `docs/business/assumptions.json` or the economics model without updating the generated outputs and running `python scripts/business_model.py --check`.

### 2.1 Every claim is stated and checked in Lean

Every technical claim an agent makes must be checkable, not trusted. This applies to issues, issue comments, pull request descriptions, review comments, comments on architectural decisions, and changes to `docs/ARCHITECTURE.md` or other design documents. It covers claims about correctness, invariants, bounds, equivalence, determinism, optimality, data movement, storage formats and protocol behaviour, and claims that code violates one of these.

1. **Cite or prove.** A claim presented as established cites a theorem in `formal/` by its full name and the commit at which `lake build --wfail` checks it, for example `Kadmium.RepOps.reducer_eq_treeSum` at `<commit>`. If no such theorem exists, the author writes one. In a pull request, the theorem lands in `formal/`, with its `Audit.lean` entry, in the same pull request. In an issue or comment, the author either links a pull request to `formal/` that adds it, or includes a complete Lean block, with no `sorry`, together with the `formal/` commit and the command that checked it with no errors or warnings (`lake env lean Claim.lean` from `formal/`). The theorem moves into `formal/` in the pull request that acts on the claim.
2. **A bug is a counterexample.** A claim that code is wrong states, in Lean, the property the code should have, and gives the failing input as a checked Lean counterexample against the model. It also includes a Rust reproduction, such as a failing test or a command, which ties the model to the code.
3. **Unproved means conjecture.** A claim that cannot be proved yet may be raised only as a conjecture. Give its exact Lean statement, label it **Conjecture**, and do not word it as established. An issue that rests on a conjecture is closed by a proof or a refutation in `formal/`.
4. **Name the link to the code.** A theorem about a model names the Rust items it models and how it is tied to them: **Proved (model)** with a differential fixture, or **Proved (Aeneas)** (see `formal/README.md`). Without that link, the claim is about the model only and must say so.
5. **Observations are not theorems.** Wall-clock speed, hardware behaviour, provider responses and other observations of the world cannot be proved in Lean. They stay **Measured**, with provenance in `docs/BENCHMARKS_2026-10.md` or a reproduction command. When a claim combines an observation with reasoning, such as "this placement is faster because link A is slower than link B", label the observation and prove the reasoning in Lean with the observation as a hypothesis.
6. **Lean never ships.** Lean source, the Lean runtime, Lake build output and Lean-generated C never enter any binary built from the Cargo workspace. No crate depends on `formal/`, no build script invokes `lake` or `lean`, and no FFI links Lean. Rust tests may read data that a Lean program generates, such as `crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt`; shipped code never reads it. The `formal` CI job fails if a Cargo manifest, build script or crate source references `formal/` or the Lean toolchain.

---

## 3. Repository map

Workspace members and their place in the dependency graph. Dependencies point downward only; a lower crate never imports a higher one.

| Crate | Role | Depends on |
| :--- | :--- | :--- |
| `kadmium-core` | Node roles, capability vectors, hardware and platform classification, shared types | nothing in-workspace |
| `kadmium-cuda` | Pure-Rust CUDA kernels compiled by cuda-oxide; host side builds everywhere, `device/` is excluded from the workspace | nothing in-workspace |
| `kadmium-npu` | Portable NPU kernels, Intel Level Zero backend | nothing in-workspace |
| `kadmium-grid` | The single execution substrate: lifecycle, units, schedules, replica catalog and staging, quorum, credits and Verde dispute settlement | core |
| `kadmium-sharding` | Asymmetric shard planner, $q^*$ placement, responsibility shifts, dataflow and pool slicing | core, grid |
| `kadmium-network` | KDM2 framing, TCP workers, USB-C delta bridges, striping, discovery | core |
| `kadmium-data` | `.ktb` tensor banks, checkpoint normalization, FTS3-style transfer scheduling | core, sharding |
| `kadmium-rl` | GRPO references, Dream-RSI replay | core, sharding |
| `kadmium-runtime` | Training and inference reference runtimes, RepOps reference math (`repops`), DiLoCo, DeMo, expert cache, kernel dispatch | core, cuda, data, grid, network, npu, sharding |
| `kadmium-condenser` | Inference workspace: providers, local checkpoints, agent harness, embedded UI | core, data |
| `kadmium-gateway` | Axum coordinator, `/v1` diagnostics, cluster API, MCP, Control Room, Condenser mount | everything above except cuda and npu directly |
| `kadmium-cli` | The `kadmium` binary | gateway and most libraries |
| `kadmium-desktop` | Native Companion (WebView2, WKWebView, optional WebKitGTK) | gateway, native |
| `kadmium-native` | Shared OS file dialogs and desktop IPC bridge; used by both native shells | nothing in-workspace |

Add a mechanism to the crate that owns it. Do not duplicate a planner, cache, budget or transport in a higher crate because it is convenient. Memory budgets, owner offers and transfer windows are meant to have one authority; see issue #60 before adding another.

What `kadmium-grid` owns: placement, staging and resource spending go through `ReplicaCatalog` and `StagingJob` in `crates/kadmium-grid/src/catalog.rs`, funded by an approved `ResourceOffer` from `kadmium-core`. A new store, transfer path or scheduler that records placements or spends memory, bandwidth or credits without them is a duplicate, however it is named.

Key documents:

- `docs/ARCHITECTURE.md` is the design reference. Section 9 covers the grid substrate, RepOps, EPLB, DSpark, KV tiers, DeepGEMM, DeepEP, Ollama kernels, CUDA, NPU and InferSSD.
- `formal/README.md` lists the Lean theorems, the Rust items each one covers and how each proof is tied to the code. Lean is the design-verification tool for algorithms, data movement and storage formats: a new or changed algorithm, transfer protocol or file format must state the property it relies on (bit-identity, bounds, rejection of malformed input, or optimality under a named cost model) and prove it there, as section 2.1 requires.
- `docs/api-reference.md` and `docs/cli-reference.md` are maintained by hand from the route handlers and clap definitions respectively. Nothing generates them. Update them in the same PR as the code.
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
| Model setup from either interface | same fixture build, then `node --test crates/kadmium-gateway/tests/model_setup.test.cjs` (arrives with PR #54) |
| CLI subcommands or Condenser engines | the CI smoke commands: `cargo run --locked --bin kadmium -- probe`, `simulate`, `dream`, `slice --pool-size 100 --slices 5`, `dataflow --model deepseek-v3-671b-moe --batch-tokens 512`, `cloud-shield`; `cargo run --locked --bin kadmium-condenser -- dsv41-profile --json` |
| Installers | `sh scripts/test_install.sh` or `./scripts/test_installers.ps1 -Executable target/debug/kadmium.exe` |
| `formal/`, or a Rust item a theorem in `formal/README.md` covers | `cd formal && lake build --wfail`, then regenerate the fixtures with `lake exe repops_vectors ../crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt` and `lake exe kadmium_vectors ..`, and run `cargo test --locked -p kadmium-runtime --test repops_lean_vectors` plus every `*_lean_vectors` test in the crates whose fixtures changed (`cargo test --locked -p <crate> -- lean_vectors`). A Rust change that alters a covered item's behaviour updates the model and its proof in the same PR. |
| Business or economics inputs | `python scripts/test_business_model.py` and `python scripts/business_model.py --check` (Python 3.12) |
| `crates/kadmium-cuda/device` | The device harness builds with `cargo oxide` on Linux or WSL2 with CUDA 13 and its own pinned toolchain. See `crates/kadmium-cuda/README.md`. It is not covered by workspace tests. |
| `kadmium-npu` hardware path | `cargo test --locked -p kadmium-npu --test intel_hardware -- --ignored` on a machine with a supported Intel NPU |

On Windows set `KADMIUM_TEST_BROWSER=msedge` for the workspace browser suite. Browser traces land in `crates/kadmium-gateway/tests/ui/test-results` and screenshots in `target/checkpoint-smoke`.

Formatting discipline: `rustfmt` on a `lib.rs` or `main.rs` follows `mod` declarations and reformats every child module. Format only files you changed, and check an untouched file through stdin (`git show HEAD:path | rustfmt --check --edition 2024`) so your diff contains only your change.

---

## 5. Code conventions

- **Pure Rust.** No Python, Node or shell at runtime for the shipped binaries. Node and Python appear only in tests and planning scripts. Lean appears only in `formal/` and never enters any binary built from the workspace (section 2.1, rule 6). Adding a dependency needs a reason in the PR body; prefer the workspace-pinned versions in the root `Cargo.toml`.
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

1. **Start from current `main`.** Fetch first; a local `main` may be stale. Create a branch named for the change, such as `feat/<topic>`, `fix/<topic>` or `docs/<topic>`. Never commit directly to `main`. If you are working in a worktree, run everything from that worktree. If your change depends on another open PR, base your branch on that PR's branch and target it, or wait for it to merge. A PR that targets `main` while carrying another open PR's commits merges that PR silently when it lands; check with `git log origin/main..HEAD` before opening.
2. **Keep the change focused.** One concern per PR. Do not bundle formatting of untouched files, unrelated refactors or drive-by doc edits.
3. **Run the checks** from section 4 that your diff touches, plus the full host sequence. Record the exact commands and results.
4. **Commit** with a conventional prefix matching the log: `feat(scope):`, `fix(scope):`, `docs:`, `test:`, `ci:`. Scope is the crate or area. The subject is imperative and under 72 characters.
5. **Push to `origin`** and **open a pull request against `main`** with `gh pr create` or the equivalent. The PR body must contain:
   - **Summary.** What changed and why, in the voice of section 5.
   - **Model quality impact.** State explicitly that no default was lowered, no truncation, compaction, quantization or fallback was added, and no advertised maximum exceeds what the backend serves. If any limit was changed, quote the new operator-facing tradeoff text.
   - **Claims.** Any number or capability statement added to docs or UI, with its Measured, Derived or Modelled label, and every claim that section 2.1 covers, with the Lean theorem and commit that prove it or its **Conjecture** label.
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
8. **Then remove your worktree.** Once the PR is merged and any follow-up issue is filed, delete the worktree you created or were assigned for the task, so finished checkouts do not pile up. Before you remove it:
   - Confirm it is yours. Never remove the main checkout (the first entry of `git worktree list`) or a worktree that another session is using.
   - Confirm nothing would be lost. `git status --porcelain` must print nothing, and `git fetch origin pull/<N>/head && git log --oneline FETCH_HEAD..HEAD` must print nothing, where `<N>` is the merged PR. A commit pushed after the merge is not in `main`. Open a new PR for it and keep the worktree until that PR merges.
   - Remove it from the main checkout, not from inside the worktree: `git -C <main checkout> worktree remove <path>`, then `git -C <main checkout> branch -D <branch>`. Use `-D` because a squash merge leaves the branch unmerged in git's view. Leave the remote branch to the reviewer. On Windows the directory can stay behind, empty, while a shell still uses it as its working directory. That is harmless.
   - If the reviewer closes the PR without merging, keep the worktree and say so in your final report. Do not delete unmerged work.
   - Remove scratch worktrees you created to review or test another PR when that review is finished, after the same checks.

Agents operating in a harness that provides PR linking or watching tools should use them so the session is notified of review activity; otherwise poll the PR at a sensible cadence until it is merged.

---

## 7. Things to never do

- Lower any model quality default, add a hidden cap, or add a fallback that changes the model, precision, context or history. See section 1.
- Present modelled, simulated or fixture output as measured, or a roadmap item as shipped.
- Commit to `main`, force-push over reviewed commits, merge your own PR, or open a PR against `main` that carries another open PR's commits.
- Weaken, skip or delete a test to get green, especially `quality_defaults.rs`, checkpoint integrity tests, digest comparisons and bridge resume tests.
- Weaken a Lean theorem, drop it from `Audit.lean`, or reach for `sorry`, `native_decide` or a new `axiom` to get a proof through.
- Reformat files you did not otherwise change.
- Remove a worktree that has uncommitted changes, holds commits that are not in `main` or a merged PR, or belongs to another session.
- Add `unwrap` or unchecked indexing on network, file or provider data.
- Widen a listener, relax same-origin or loopback checks, or log secrets.
- Add emoji, exclamation marks, marketing adjectives, or the reactor metaphor beyond the one tagline.
- Describe `/v1` gateway routes as real LLM generation, or Kadmium Cloud, payouts, procurement or settlement as live.
- Open an issue, post a review or architecture comment, or write a pull request description whose claims are not backed by Lean as section 2.1 requires.
- Build, link, embed or invoke Lean from anything compiled from the Cargo workspace.
