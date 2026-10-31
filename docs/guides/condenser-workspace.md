# Condenser inference workspace

Condenser connects conversations and tool runs to inference providers while keeping Kadmium's compute plans within reach. Open **Models → Launch Condenser** in Kadmium's Control Room or desktop Companion. **Prepare a local model** opens GGUF/SafeTensors setup. Condenser is included at `/condenser/` and stays inside the desktop window. A separate launch is optional:

```sh
cargo run -p kadmium-condenser -- serve --port 7856
```

Open `http://127.0.0.1:7856`. Configure the task's provider, base URL and model using **Provider settings**. Kadmium currently returns reference diagnostics; Ollama, OpenAI-compatible, Anthropic and MCL4-COE adapters use the endpoint you configure. Selecting Kadmium never silently switches the task to another provider.

## Inspect compute alongside a task

1. Start the Kadmium coordinator separately with `kadmium serve`.
2. Open the gold compute icon in Condenser's workspace dock. Enter the coordinator URL (normally `http://127.0.0.1:7848`) and a coordinator credential if your deployment requires one.
3. Choose **Inspect compute** to read the fleet, planning catalog and current placement. Select a catalog model and inspect again to preview another placement.
4. Choose **Attach snapshot to task** to add the timestamped plan, node details and qualifications to the next prompt. This sends the snapshot to the task's chosen inference provider when you run it.

The preview does not select, load or deploy a model. It distinguishes live hardware from demo-plus-live fleets, counts registered online workers, and shows each node's layers and planned memory placement. The underlying planner estimates placement and throughput; its output does not establish that model weights are available or generation can run. **Open control room** opens the inspected coordinator for operator actions.

Changing the coordinator URL or planning model invalidates the displayed snapshot. Inspection has an eight-second timeout and a 2 MiB response limit. Reverse-proxy path prefixes are supported; redirects are not followed. The coordinator credential is independent of inference-provider credentials and is never attached to a prompt.

## Load checkpoints and run distributed jobs

Choose **Models & compute** in the sidebar or **Load a local checkpoint** in Provider settings. Normalize a GGUF file or complete SafeTensors directory, inspect its source precision and context, then load it in local Ollama. **Use in this task** explicitly selects the loaded model; a failed import or load leaves task routing intact. See [local checkpoints](local-checkpoints.md) for compatibility and the lossless storage contract.

The same panel provides worker connection instructions and **Run on workers** for supplied FP32 matrix products, reference training previews, or replay. Jobs require at least two trusted online TCP workers and never fall back to demo or in-process execution. Results include actual node IDs, measured wall time, unit traces and failures. The training preview does not train an imported checkpoint. Follow [distributed compute](distributed-compute.md) to run jobs or synchronize real learners with the existing stateful DiLoCo API.

## Continue and control work

- Draft prompts and imported text attachments stay with their task when switching tasks or reloading the same browser origin.
- **Fork** copies a stopped task's conversation, run history, draft, instructions and settings. The fork has independent state and records its parent task. It does not copy external provider sessions or compute allocations.
- The **Run** dock exposes time, aggregate output-token and tool-call limits. Empty values mean no limit; zero tool calls prevents tool execution. Output limits are checked between model steps, so a single response may exceed the requested total.
- **Stop** cancels the active provider/tool wait and retains the final report with completed steps. It cannot undo completed tool side effects. Partial streamed text appears in the runbook and is not replayed as a completed assistant turn.
- Every new run records its provider, model and base URL. The run inspector shows the last route; exports retain all run routes.

An interrupted stream never automatically resubmits a request. Partial output remains in the run history, and the user chooses how to continue. Closing the stream cancels the server turn, but a closed browser cannot receive its final report.

Tasks, forks, drafts and exports remain browser-local. API keys are memory-only and inference keys are scoped to both provider and endpoint. Changing endpoints does not carry a key to the new server. Browser storage limits can still prevent saving large histories; the sidebar shows `TEMP` when persistence fails.

## Queue follow-ups

While a normal task runs, the composer changes to **Queue follow-up**. Add up to 20 prompts, including text attachments, without interrupting the current response. The queue runs in order after each successful response, using the conversation accumulated so far. Each item captures its provider, model, endpoint, instructions, tool selection, sampling and run limits when submitted. Changing task settings later does not change queued items. API keys are resolved from memory for the captured endpoint when execution starts.

Use **Edit**, the up arrow, or remove to revise the queue. Editing pauses it; **Resume queue** continues after the active run, and **Run queue** starts it when idle. **Pause queue** leaves the current run alone. Stop, provider errors, exhausted budgets and interrupted streams pause remaining items. Reloading always restores the queue paused, and an interrupted run is marked for review instead of replayed. An item that already started remains in conversation/run history, not in the pending queue.

Queues belong to their task. Switching tasks preserves the active task's queue and both composer drafts; completed background runs do not switch your task. Forks start with an empty queue. Q/A interviews and follow-up queues are separate workflows: finish Q/A before queuing follow-ups. This is a browser-managed queue, not a background scheduler; the workspace must remain open to advance it.

## Revisit a prompt

Choose **Edit & branch** beside a user prompt to create an independent task with the conversation before that prompt. The selected prompt becomes an editable draft, including any text already attached to it. Later messages and queued actions stay in the original task. Branches cut at user turns so previous assistant tool calls remain paired with their results.

Runs recorded by this version carry conversation boundaries, allowing the branch to retain only preceding run records. Older run records without boundaries stay in the original task. Regular **Fork** still copies the entire accepted conversation and its run history.

## Select tools per task

The **Run → Tool access** inspector lists discovered tools as individual checkboxes, with **Select all** and **Clear** controls. The first task starts with an empty selection; subsequent new tasks inherit the current task's settings. Turn **Tools on** and select the capabilities needed for the task. Tool selections persist with tasks and forks, and queued prompts capture their own selection. An unavailable selected tool is shown in the inspector and must be removed or restored before running with tools enabled.

The runtime filters both the model's tool catalog and the execution registry. A model request for an excluded tool receives a paired error result without starting the plugin. Tool selection limits model-initiated calls; explicit **Test tool** invocations in the registry remain direct user actions. Existing API clients and older tasks without a selection retain their previous all-discovered-tools behavior until an explicit list is saved.

CLI callers can restrict tool access with repeated exact names:

```sh
cargo run -p kadmium-condenser -- chat "Inspect the project" --provider ollama --tools --allow-tool files__read --allow-tool files__list
```

Those names must exist in the local plugin registry; they are examples, not built-in tools.

## Maximum-quality defaults

**Provider settings → Quality & token limits** defaults to full model context/input, maximum output and no generation timeout. Explicitly lower **Context / input window** for Ollama, **Output tokens per response**, or **Request timeout** when needed. **Reset to maximum** clears these overrides; Apply saves them per task. Forks inherit them and queued prompts retain settings captured at submission. Existing explicit user limits remain intact. Input and output share a context budget, so both cannot occupy the entire window simultaneously.

Ollama resolves native context from model metadata on every request, disables prompt truncation and uses unlimited-until-stop generation. Other providers receive the complete conversation; context allocation is configured on their servers. Anthropic resolves its advertised output maximum from model metadata; OpenAI-compatible and MCL4 endpoints resolve it from their model catalog. If required metadata is absent, generation stops and asks for the model's documented output limit. Condenser never substitutes 1,024 tokens or another guessed cap. Hosted APIs that do not publish limits therefore need an explicit output setting.

MCL4 history compaction and old tool-result pruning are off by default and require **Allow history compaction**. Temperature remains the provider default because higher temperature does not mean higher quality. Run-level budgets remain separate explicit controls. Allocation and precision failures do not substitute another model or quantize weights. Backend numerical precision and tokenization remain backend responsibilities.

## API contracts

Kadmium exposes `GET /api/v1/inference/workspace?model_id=<catalog-id>`. Omitting `model_id` previews the coordinator's current model. Unknown IDs return HTTP 400 instead of selecting a fallback. The version 1 response includes `epoch`, `fleet_mode`, `generation_supported`, `selected_model_id`, `online_workers`, `catalog`, `nodes`, `plan`, `assigned_layers` and `warnings`. This endpoint is read-only and does not increment the cluster epoch.

Condenser exposes `POST /api/v1/compute/inspect` with `base_url`, optional `api_key`, and optional `model_id`; it returns the coordinator snapshot. Like Condenser's other APIs, it requires loopback Host/Origin access.

Checkpoint endpoints and the compute job proxy are documented in the [API reference](../api-reference.md#condenser-checkpoints-and-compute-jobs). Importing, backend loading and running jobs are explicit writes, separate from the read-only placement preview.

Both `POST /api/v1/chat` and `POST /api/v1/chat/stream` accept an optional `run_id` of 1–80 ASCII letters, digits, hyphens or underscores. IDs must be unique among active runs. `POST /api/v1/runs/{run_id}/cancel` returns `{"status":"cancelling","run_id":"..."}` or HTTP 404 if the run is no longer active. Cancellation completion is reported by the original request, not the cancellation acknowledgement. A run ID is not an idempotency key and may be reused after a run ends.

Both chat endpoints also accept `allowed_tools`: omit it or use `null` for all discovered tools, use `[]` for none, or supply exact qualified names such as `["files__read"]`. `enable_tools: false` always disables model tool execution. With tools enabled, unknown or unqualified list entries fail before inference. Streaming reports that configuration failure as an `error` event; nonstreaming requests return HTTP 400. The allowlist does not accept wildcards or argument patterns.

Streaming ends with one authoritative `finished` event containing the full harness report, or an `error` event if no report can be produced. Progress events may be dropped under backpressure; the final report is retained. Existing clients can omit `run_id` and cancel streaming runs by closing the response body.

## Development checks

```sh
cargo test -p kadmium-condenser -p kadmium-gateway
node --test crates/kadmium-condenser/tests/workflow.test.cjs
```

The workspace follows T3's conversation/sidebar/inspector and editable queue patterns, DeepSeek harness's explicit run completion and queue cancellation behavior, and Claude Code's scoped tool access. The implementation uses the existing Rust server and embedded JavaScript UI. Local model and compute controls use Kadmium's shared graphite and cadmium design tokens. See [contributing](../../CONTRIBUTING.md) for the additional browser workflow check.
