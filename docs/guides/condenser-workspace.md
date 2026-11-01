# Condenser inference workspace

In the desktop apps, **Open file** in the Files panel uses the system file dialog. Select a file inside the active workspace; the existing workspace and sensitive-file checks still apply. The editor, autosave and diff stay in Condenser. **Show workspace tree** opens the embedded tree when you need it alongside the editor. Browser sessions use that tree by default. Attachment import already uses the browser or WebView's system file chooser. Native Linux dialogs use the desktop portal, with the dialog library's Zenity fallback when the portal is unavailable. If no system dialog opens, use the workspace tree or the checkpoint chooser's **Use embedded browser** action.

Condenser connects conversations and tool runs to inference providers while keeping Kadmium's compute plans within reach. Open **Models → Launch Condenser** in Kadmium's Control Room or desktop Companion. **Prepare a local model** opens GGUF/SafeTensors setup. Condenser is included at `/condenser/` and stays inside the desktop window.

Because Condenser can read and write files and run commands, each coordinator launch has its own access link. The desktop app opens it for you. `kadmium serve` prints it as **Condenser workspace** and saves it in `condenser-access-<port>` in the Kadmium state directory, readable only by your account. Opening the link stores an HttpOnly cookie for `/condenser`. The link is reusable until the coordinator restarts, so treat it like a password for that launch. Requests without it get HTTP 401, even from loopback, and requests from LAN peers or other origins are refused. The embedded workspace's file, terminal, agent-tool and plugin root is `condenser-workspace` in that state directory, never the coordinator's working directory. Set `KADMIUM_CONDENSER_WORKSPACE` to an absolute directory to choose another root. **Workspace** shows the active root.

A separate launch is optional; standalone Condenser uses the directory it is started from:

```sh
cargo run -p kadmium-condenser-app -- serve --port 7856
```

Open `http://127.0.0.1:7856`. Configure the task's provider, server address and model using **Model & quality** (the provider button in the top bar). Kadmium currently returns reference diagnostics; Ollama, OpenAI-compatible, Anthropic and MCL4-COE adapters use the endpoint you configure. Selecting Kadmium never silently switches the task to another provider.

## Inspect compute alongside a task

1. Start the Kadmium coordinator separately with `kadmium serve`.
2. Open the gold compute icon in Condenser's workspace dock. Enter the coordinator URL (normally `http://127.0.0.1:7848`) and a coordinator credential if your deployment requires one.
3. Choose **Inspect compute** to read the fleet, planning catalog and current placement. Select a catalog model and inspect again to preview another placement.
4. Choose **Attach snapshot to task** to add the timestamped plan, node details and qualifications to the next prompt. This sends the snapshot to the task's chosen inference provider when you run it.

The preview does not select, load or deploy a model. It distinguishes live hardware from demo-plus-live fleets, counts registered online workers, and shows each node's layers and planned memory placement. The underlying planner estimates placement and throughput; its output does not establish that model weights are available or generation can run. **Open control room** opens the inspected coordinator for operator actions.

Changing the coordinator URL or planning model invalidates the displayed snapshot. Inspection has an eight-second timeout and a 2 MiB response limit. Reverse-proxy path prefixes are supported; redirects are not followed. The coordinator credential is independent of inference-provider credentials and is never attached to a prompt.

## Load checkpoints and run distributed jobs

Choose **Models** in the sidebar or **Bring your own model…** in Model & quality. Prepare a GGUF file or complete SafeTensors folder, inspect its precision and context, then run it on **This computer** through Ollama, send it to a **Remote coordinator** you run, or choose **Split across my machines** to plan contiguous decoder stages across your coordinator's machines and inspect which machine holds which layers, embeddings, final norm, LM head and full-context KV cache. **Use in current task** explicitly selects a loaded model; a failed preparation, load, send or split plan leaves task routing intact. See [local checkpoints](local-checkpoints.md) for compatibility and the lossless storage contract.

The compute dock (chip icon beside the dock tabs) provides worker connection instructions and **Run on workers** for supplied FP32 matrix products, reference training previews, or replay. Jobs require at least two trusted online TCP workers and never fall back to demo or in-process execution. Results include actual node IDs, measured wall time, unit traces and failures. The training preview does not train an imported checkpoint. Follow [distributed compute](distributed-compute.md) to run jobs or synchronize real learners with the existing stateful DiLoCo API.

## Continue and control work

- Draft prompts and imported text attachments stay with their task when switching tasks or reloading the same browser origin.
- **Fork** copies a stopped task's conversation, run history, draft, instructions and settings. The fork has independent state and records its parent task. It does not copy external provider sessions or compute allocations.
- The **Run** dock exposes time, aggregate output-token and tool-call limits. Empty values mean no limit; zero tool calls prevents tool execution. Output limits are checked between model steps, so a single response may exceed the requested total.
- **Stop** cancels the active provider/tool wait and retains the final report with completed steps. It cannot undo completed tool side effects. Partial streamed text appears in the runbook and is not replayed as a completed assistant turn.
- Every new run records its provider, model and base URL. The run inspector shows the last route; exports retain all run routes.

An interrupted stream never automatically resubmits a request. Partial output remains in the run history, and the user chooses how to continue. Closing the stream cancels the server turn, but a closed browser cannot receive its final report.

Tasks, forks, drafts and exports remain browser-local unless **Save task memory for workspace recall** is enabled. That option also retains full run reports on the server's local disk, as described below. API-key configuration is memory-only and inference keys are scoped to both provider and endpoint. Changing endpoints does not carry a key to the new server. Browser storage limits can still prevent saving large histories; the sidebar shows `TEMP` when persistence fails.

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
cargo run -p kadmium-condenser-app -- chat "Inspect the project" --provider ollama --tools --allow-tool files__read --allow-tool files__list
```

Those example plugin names must exist in the local plugin registry. First-party tools are available without installing plugins under the reserved `condenser` namespace.

## Built-in agent tools

Enable **Tools** and select individual capabilities under **Run → Tool access**. New tasks still start with an empty selection. API clients that omit `allowed_tools` while enabling tools expose discovered plugin tools and the read-only built-ins. The side-effecting built-ins (`condenser__command`, `condenser__file_edit`, `condenser__web_fetch`, `condenser__delegate`) are never part of "all". They are available only when named in `allowed_tools` (or checked individually in the UI), and the report lists any that were withheld. `condenser__command` also needs the owner's confirmation for the task: **Allow host commands** in the UI, `agent.allow_host_commands: true` in the API, or `--allow-host-commands` in the CLI. Without it the run fails before inference.

| Qualified name | Capability |
| --- | --- |
| `condenser__files_list` | Sorted directory listing, 200 entries per page. |
| `condenser__file_read` | UTF-8 file content, 16,000 characters per page. |
| `condenser__file_search` | Literal search within a file, line numbers and 50 matches per page. |
| `condenser__file_edit` | Replace one exact, unique text span in an existing file. |
| `condenser__command` | Existing host terminal execution, with exit status and explicit timeout/truncation flags. |
| `condenser__web_fetch` | HTTP(S) source retrieval with full text saved for subsequent reads. |
| `condenser__notes_write` | Append working notes, decisions, evidence and lookup entries. |
| `condenser__plan_update` | Append a checklist/plan revision without overwriting prior versions. |
| `condenser__memory_search` | Search a local word-prefix index; return record references and provenance. |
| `condenser__memory_read` | Retrieve exact records by ID, with character offsets and `next_offset`. |
| `condenser__subagents_list` | Discover named, operator-configured local and API targets. |
| `condenser__delegate` | Execute a self-contained brief on one named target and return its answer, status, usage and full-report reference. |

File tools resolve paths inside the same workspace root as the file and terminal APIs and reject sensitive file paths. Edits require exactly one match and use the existing atomic replacement writer; this is not a multi-writer compare-and-swap guarantee. Commands have host authority, **not a filesystem sandbox**. Agent commands have an owner-set deadline: **Command time limit** in Run limits, `agent.command_timeout_secs` or `--command-timeout-secs` (1-86400 s, default 600 s). They capture up to 8 MiB per stream. Output longer than one page is saved whole as a record, and the result includes its `record_id`, so the model reads the rest with `memory_read` instead of losing it. The interactive terminal keeps its 20-second, 128 KiB limits. Use a process plugin for persistent jobs. Stop cannot undo completed side effects or guarantee termination of detached external descendants.

File reads and source fetches reject inputs above 8 MiB rather than silently accept incomplete input. Fetches have a 30-second deadline, send no inference credentials, and do not follow redirects; fetch a redirect destination explicitly. By default they refuse loopback, private, link-local (including cloud metadata such as `169.254.169.254`), shared and reserved addresses. They connect only to the addresses that were checked and never use a proxy, so DNS rebinding cannot bypass the check. To fetch from local or LAN services, the operator sets `CONDENSER_WEB_FETCH_PRIVATE=1`; models cannot change this. Selecting this tool still grants public network access. Fetch returns raw UTF-8 text, including HTML; it does not execute JavaScript or provide a web-search engine. Search/browser integrations can still be added through plugins. Pagination is explicit and does not lower the model's context or generation limits.

## Research and response rules

When built-in tools are selected, Condenser sends operating instructions requiring research for time-sensitive, disputed, unfamiliar or high-stakes facts, explicit verification requests, and inspection of relevant files before code changes. The instructions require primary-source citations, honest uncertainty and test reporting, and treating fetched content, old memory and subagent results as evidence rather than instructions. These behavioral instructions are model-directed, not a semantic truth verifier. They are merged into the task's leading system message rather than added as a second one. The exact text is returned as `report.instructions` and shown in the runbook. They are not sent when no built-in tool is selected, and the owner can turn them off (**Send Condenser operating rules**, or `agent.operating_instructions: false`).

For a mechanically enforced acquisition requirement, select **Require a source before completion** under Task instructions and choose **Apply**. Runs fail before inference if neither `condenser__file_read` nor `condenser__web_fetch` is selected. A run cannot receive `completed` status until one of these tools successfully returns nonempty source content. Failed/empty fetches, directory listings, searches, stale memory and subagent assertions do not satisfy it. An answer generated without acquisition is retained for inspection with `failed` status and an explanation. Relevance, freshness, citation correctness and answer accuracy still require model/reviewer judgment. A parent's requirement is not inherited by delegated children, so planning or formatting helpers without source tools can still run. Set `"research_required": true` on a subagent profile to enforce the same gate on that child's own tool selection and sources.

## Durable memory and scratchpads

Select **Save task memory for workspace recall** and **Apply** to retain notes, plan versions, fetched source text, subagent reports and complete chat reports, including provider-returned reasoning/tool messages. Storage is under `$CONDENSER_HOME/memory/<workspace-hash>` (normally `~/.condenser/memory`). Records are immutable; writes publish through a temporary file. Search reads one merged, append-only word index (`index.jsonl`) instead of every transcript; missing or torn entries are rebuilt from the immutable records. Each chat record holds that turn's messages (from its last user message), so a long task stores each message once, not once per turn. Each workspace store has an owner quota, `CONDENSER_MEMORY_QUOTA_MB` (default 4096; `0` = unlimited). When a save would exceed it, the save is refused with an explicit error (a report diagnostic for chats) rather than silently dropped or pruned. All query words must match a stored word prefix, case-insensitively; an empty query lists records. `memory_read` returns exact content with explicit continuation offsets. Search is lexical, not semantic/vector search.

Saved records are scoped to the canonical workspace path. Tools can recall other opted-in tasks in that workspace, including older task snapshots. The model must check provenance and revalidate stale information. Retrieval never replaces, summarizes or prunes the current conversation automatically. Forks retain the memory preference but use their own task ID; queued prompts capture the preference and research requirement at submission.

Without retention, notes and source records exist only during the current run, shared with its children. They disappear afterwards. Provider credential configuration is not part of stored reports, but prompts and tool output may contain sensitive information: enable retention only for material you intend to keep. Turning retention off prevents future writes and hides durable recall; it does not erase previous records. Remove the workspace's memory directory while runs are stopped to erase it. This is local storage, not encrypted storage or a multi-user authorization boundary. Storage failures appear in report diagnostics instead of silently claiming successful persistence.

## Configure API or local subagents

Create `~/.condenser/subagents.json`, or set `CONDENSER_SUBAGENTS_FILE` to an operator-controlled JSON file. This file is read when `subagents_list` or `delegate` runs, so a malformed file affects only those calls, not runs that never delegate. Models can select a named target but cannot supply a new endpoint, credential or model. Example:

```json
{
  "local-review": {
    "provider": "ollama",
    "model": "your-installed-model",
    "base_url": "http://127.0.0.1:11434",
    "allowed_tools": ["condenser__file_read", "condenser__file_search"],
    "max_steps": 32
  },
  "api-research": {
    "provider": "openai",
    "model": "your-api-model-id",
    "base_url": "https://your-provider.example/v1",
    "api_key_env": "RESEARCH_MODEL_API_KEY",
    "allowed_tools": ["condenser__web_fetch"],
    "max_steps": 32
  }
}
```

Supported targets use Condenser's provider adapters: Ollama, OpenAI-compatible, Anthropic and MCL4-COE. Kadmium remains a reference diagnostic adapter. InferSSD can be addressed through its OpenAI-compatible endpoint; this does not modify its engine or expert routing. An unavailable model, credential or required limit fails explicitly; there is no substitution.

Omit `sampling` and `timeout_secs` for native maximum context/output and no request timeout. Anthropic targets that do not advertise a maximum require an explicit documented `sampling.max_tokens`, as with the parent. Explicit `sampling.context_window`, `sampling.max_tokens`, `sampling.compact_history` and `timeout_secs` carry the same quality tradeoffs as parent settings. A smaller model or reduced context/output can weaken results; no cost-saving target is chosen automatically.

Delegation is synchronous and runs one child at a time. The child receives the supplied brief and operating rules, not an automatic copy of the parent history; include all necessary context or enable relevant recall tools. Its tools are the intersection of profile and parent selections, with nested delegation removed. Parent cancellation and deadlines apply, and completed child usage/tool calls count against parent budgets. Aggregate output budgets are checked between responses, so one response can exceed a budget. Partial/in-flight provider billing may not be reported after cancellation. Full answers are returned without automatic summarization; the full child report is also retrievable by record ID.

CLI example:

```sh
cargo run -p kadmium-condenser-app -- chat "Inspect Cargo.toml and explain this project" --provider ollama --tools --allow-tool condenser__file_read --allow-tool condenser__notes_write --memory-session project-review --research-required
```

`plugin list` includes the built-ins. `plugin invoke condenser subagents_list` lists configured targets without exposing keys. Direct registry invocations are user actions; a direct delegate has no child tool permissions. For multi-step memory tests through the HTTP invocation endpoint, supply `agent.session_id`; without it each direct invocation is ephemeral.

## Maximum-quality defaults

**Model & quality → Quality & limits** defaults to full model context, maximum output and no time limit, and its summary reads **Maximum quality** until you change something. Lower **Context window** (Ollama), **Longest response**, or **Time limit per request** only when needed; each field explains the tradeoff as you type and the summary lists every reduction. **Restore maximum quality** clears these overrides; Apply saves them per task. Forks inherit them and queued prompts retain settings captured at submission. Existing explicit user limits remain intact. Input and output share a context budget, so both cannot occupy the entire window simultaneously.

Ollama resolves native context from model metadata on every request, disables prompt truncation and uses unlimited-until-stop generation. Other providers receive the complete conversation; context allocation is configured on their servers. Anthropic requires `max_tokens`, so Condenser resolves its advertised output maximum from model metadata. If that is absent, generation stops and asks for the model's documented output limit. OpenAI-compatible endpoints and the MCL4 OpenAI-compatible fallback send an output limit only when their model catalog explicitly advertises one (`max_output_tokens`, `max_completion_tokens` or `top_provider.max_completion_tokens`). Otherwise they omit `max_tokens`, which these APIs treat as the server's maximum. Context-window fields are never read as an output limit. The native MCL4 API is not preflighted. Condenser never substitutes 1,024 tokens or another guessed cap.

MCL4 history compaction and old tool-result pruning are off by default and require **Compact long histories**. Temperature remains the provider default because higher temperature does not mean higher quality. Run-level budgets remain separate explicit controls. Allocation and precision failures do not substitute another model or quantize weights. Backend numerical precision and tokenization remain backend responsibilities.

Chat HTTP bodies bypass the framework's generic 2 MiB request limit, and successful generation responses bypass the 16 MiB metadata-response cap. The complete conversation and response can therefore use the model's supported limits, subject to available host memory and upstream server/proxy configuration. Explicit user token/run limits remain effective; metadata, plugin, file and fetch limits are separate from model generation.

## API contracts

Kadmium exposes `GET /api/v1/inference/workspace?model_id=<catalog-id>`. Omitting `model_id` previews the coordinator's current model. Unknown IDs return HTTP 400 instead of selecting a fallback. The version 1 response includes `epoch`, `fleet_mode`, `generation_supported`, `selected_model_id`, `online_workers`, `catalog`, `nodes`, `plan`, `assigned_layers` and `warnings`. This endpoint is read-only and does not increment the cluster epoch.

Condenser exposes `POST /api/v1/compute/inspect` with `base_url`, optional `api_key`, and optional `model_id`; it returns the coordinator snapshot. Like Condenser's other APIs, it requires loopback Host/Origin access.

Checkpoint endpoints and the compute job proxy are documented in the [API reference](../api-reference.md#condenser-checkpoints-and-compute-jobs). Importing, backend loading and running jobs are explicit writes, separate from the read-only placement preview.

Both `POST /api/v1/chat` and `POST /api/v1/chat/stream` accept an optional `run_id` of 1–80 ASCII letters, digits, hyphens or underscores. IDs must be unique among active runs. `POST /api/v1/runs/{run_id}/cancel` returns `{"status":"cancelling","run_id":"..."}` or HTTP 404 if the run is no longer active. Cancellation completion is reported by the original request, not the cancellation acknowledgement. A run ID is not an idempotency key and may be reused after a run ends.

Both chat endpoints also accept `allowed_tools`: omit it or use `null` for all discovered tools, use `[]` for none, or supply exact qualified names such as `["files__read"]`. `enable_tools: false` always disables model tool execution. With tools enabled, unknown or unqualified list entries fail before inference. Streaming reports that configuration failure as an `error` event; nonstreaming requests return HTTP 400. The allowlist does not accept wildcards or argument patterns.

Both chat endpoints accept `agent: {"session_id": "task-id", "research_required": true}`. Both fields are optional: no session means no durable retention, and research acquisition is not mechanically gated unless requested. Session IDs allow 1–100 ASCII letters, digits, hyphens and underscores. `POST /api/v1/plugins/invoke` also accepts `agent` for explicit built-in invocations. No credential/profile configuration is accepted from model tool arguments.

Streaming ends with one authoritative `finished` event containing the full harness report, or an `error` event if no report can be produced. Progress events may be dropped under backpressure; the final report is retained. Existing clients can omit `run_id` and cancel streaming runs by closing the response body.

## Development checks

```sh
cargo test -p kadmium-condenser -p kadmium-gateway
node --test crates/kadmium-condenser/tests/workflow.test.cjs
```

The workspace follows T3's conversation/sidebar/inspector and editable queue patterns, DeepSeek harness's explicit run completion and queue cancellation behavior, and Claude Code's scoped tool access. The implementation uses the existing Rust server and embedded JavaScript UI. Local model and compute controls use Kadmium's shared graphite and cadmium design tokens. See [contributing](../../CONTRIBUTING.md) for the additional browser workflow check.

The agent tools also draw on T3's explicit delegation targets and capability inheritance, DeepSeek harness's session-query separation from compaction, and InferSSD's immutable indexes and quality-preserving optimization boundaries. This is a harness change, not an InferSSD kernel integration or a measured intelligence/throughput claim. Regression tests verify history, token-limit, routing, cancellation and recall contracts; model-quality improvements require representative task evaluations.
