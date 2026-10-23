# Launch Kadmium and inspect the setup plan

`kadmium launch` is an additive convenience command for the existing coordinator. It probes this machine, creates a configuration only if it is missing, and starts the Control Room. The current `/v1` endpoints exercise reference compute and return diagnostic output; launching does not download weights, start a language model or connect an external inference server.

## Inspect before starting

```bash
kadmium launch --plan
kadmium launch --plan --config lab.toml --port 8848
```

The plan is human-readable. It reports the probed hardware, an informational planning preset, configuration path, HTTP port and worker port. It performs no file writes, network listener startup, bridge discovery or model download. A recommendation is a planning estimate, not proof that a model fits or can generate text. The coordinator makes its own recommendation when it starts.

## Start the existing coordinator

```bash
kadmium launch
kadmium launch --config lab.toml --port 7848
```

The default configuration path is `kadmium.toml`. An existing readable file is preserved byte for byte. A missing file is generated from the host probe and written with exclusive creation so another file cannot be overwritten if it appears during startup. There is no force-overwrite option on `launch`.

The existing configuration parser and coordinator behavior are unchanged. The coordinator reads the file's CloudShield settings and workspace name for that mode; the file does not override the listener, host capabilities or active model. `--port` controls the HTTP port. A generated model recommendation is informational and is not a fixed model selection for the running coordinator.

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--plan` | off | Print the plan and exit without starting or writing. |
| `-c`, `--config <PATH>` | `kadmium.toml` | Preserve an existing file or create a missing one. |
| `-p`, `--port <PORT>` | `7848` | HTTP listener port. |
| `--worker-port <PORT>` | `7850` | Preferred built-in worker TCP port; `0` asks the OS for a free port. |
| `--demo` | off | Include the simulated seven-node fleet. |
| `--spawn-fleet` | off | Start the existing local worker test fleet. |

The coordinator and built-in worker listen on all interfaces, as they do with `serve`. Startup retains the existing LAN discovery and hardware bridge discovery behavior. Use this on a trusted network and use firewall or reverse-proxy access controls as needed; an inference-shaped URL or a client API key is not an authentication boundary. This change does not alter `serve`, `desktop`, their defaults or `init` overwrite behavior.

Open `http://127.0.0.1:7848` for the Control Room. A peer can join with `kadmium join --coordinator http://<coordinator-lan-ip>:7848`; automatic discovery uses Kadmium UDP announcements on port `7849`, not mDNS.

## Inspect the reference API

```bash
curl http://127.0.0.1:7848/v1/models
curl http://127.0.0.1:7848/api/v1/grid/inventory
curl -X POST http://127.0.0.1:7848/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Run a reference step."}]}'
```

The model list is a planning catalog. Chat responses contain fixed diagnostic text and execution telemetry; embeddings are reference vectors. These are not general OpenAI or Anthropic SDK replacements. See the [API reference](../api-reference.md) for the request behavior and limitations.

## Dependency decision

This setup change **keeps the existing dependency set**: no Cargo manifest or lockfile changes and no new direct or transitive Rust dependencies. An external inference relay is deferred, with no optional relay implementation included. Adding one requires a future owner decision about dependencies and downstream client access, authentication, streaming and resource limits.

The setup workflow takes inspiration from the reviewed [Strata revision](https://github.com/niko1221/strata/tree/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec). Kadmium's current diagnostic routes do not inherit Strata's runtime capabilities. See [installation](../getting-started.md#1-installation) for the release archive and checksum workflow.
