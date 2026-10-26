(function () {
  'use strict';

  const STORAGE_KEY = 'condenser.agent.tasks.v1';
  const FALLBACK_PROVIDERS = {
    kadmium: { label: 'Kadmium', base: 'http://127.0.0.1:7848', model: 'deepseek-v3-671b-moe-q4k', keyEnv: 'KADMIUM_API_KEY', note: 'Kadmium currently provides reference diagnostics, not model-scale generation.' },
    ollama: { label: 'Ollama', base: 'http://127.0.0.1:11434', model: 'llama3.2', keyEnv: 'OLLAMA_API_KEY', note: 'Use models already available from a local or remote Ollama daemon.' },
    openai: { label: 'OpenAI-compatible', base: 'https://api.openai.com', model: '', keyEnv: 'OPENAI_API_KEY', note: 'Use a hosted OpenAI-compatible API or a local server such as LM Studio or llama.cpp.' },
    anthropic: { label: 'Anthropic', base: 'https://api.anthropic.com', model: '', keyEnv: 'ANTHROPIC_API_KEY', note: 'Use the Anthropic Messages API with a request-scoped API key.' }
  };

  const state = {
    providers: [],
    plugins: [],
    sessions: [],
    currentId: null,
    credentials: Object.create(null),
    busy: false,
    busySessionId: null,
    selectedTool: null,
    persistence: true
  };

  const $ = (id) => document.getElementById(id);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function uid() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
    return 'task-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = {};
    if (text) {
      try { body = JSON.parse(text); } catch (_) { body = { error: text }; }
    }
    if (!response.ok) throw new Error(body.error || ('HTTP ' + response.status));
    return body;
  }

  function descriptor(kind) {
    const remote = state.providers.find((item) => item.kind === kind);
    const fallback = FALLBACK_PROVIDERS[kind] || FALLBACK_PROVIDERS.kadmium;
    return {
      kind,
      label: remote ? remote.label : fallback.label,
      default_base_url: remote ? remote.default_base_url : fallback.base,
      default_model: remote ? remote.default_model : fallback.model,
      default_api_key_env: remote ? remote.default_api_key_env : fallback.keyEnv,
      requires_api_key: remote ? remote.requires_api_key : kind === 'anthropic',
      note: remote ? remote.note : fallback.note
    };
  }

  function defaultConfig(kind) {
    const item = descriptor(kind || 'kadmium');
    return {
      kind: item.kind,
      model: item.default_model,
      base_url: item.default_base_url,
      enable_tools: false,
      max_steps: 8,
      system_prompt: '',
      sampling: {}
    };
  }

  function normalizeConfig(raw) {
    const kind = raw && FALLBACK_PROVIDERS[raw.kind] ? raw.kind : 'kadmium';
    const base = defaultConfig(kind);
    const config = Object.assign(base, raw || {});
    config.kind = kind;
    config.max_steps = Math.max(1, Math.min(32, Number(config.max_steps) || 8));
    config.enable_tools = Boolean(config.enable_tools);
    config.system_prompt = typeof config.system_prompt === 'string' ? config.system_prompt : '';
    config.sampling = config.sampling && typeof config.sampling === 'object' ? config.sampling : {};
    return config;
  }

  function normalizeSession(raw) {
    if (!raw || typeof raw !== 'object' || !raw.id) return null;
    return {
      id: String(raw.id),
      title: typeof raw.title === 'string' && raw.title.trim() ? raw.title : 'Untitled task',
      createdAt: Number(raw.createdAt) || Date.now(),
      updatedAt: Number(raw.updatedAt) || Date.now(),
      messages: Array.isArray(raw.messages) ? raw.messages.filter((message) => message && typeof message === 'object') : [],
      runs: Array.isArray(raw.runs) ? raw.runs.filter((run) => run && typeof run === 'object') : [],
      config: normalizeConfig(raw.config),
      status: ['ready', 'complete', 'error'].includes(raw.status) ? raw.status : 'ready',
      lastError: typeof raw.lastError === 'string' ? raw.lastError : null
    };
  }

  function currentSession() {
    return state.sessions.find((session) => session.id === state.currentId) || null;
  }

  function loadSessions() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
      if (stored && Array.isArray(stored.sessions)) {
        state.sessions = stored.sessions.map(normalizeSession).filter(Boolean);
        state.currentId = stored.currentId;
      }
    } catch (error) {
      console.warn('Condenser could not restore task history:', error);
    }
    if (!state.sessions.length) createTask(false);
    if (!state.sessions.some((session) => session.id === state.currentId)) {
      state.currentId = state.sessions.slice().sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
    }
    saveSessions();
  }

  function saveSessions() {
    try {
      const sessions = state.sessions.map((session) => Object.assign({}, session, {
        status: session.status === 'running' ? 'ready' : session.status
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, currentId: state.currentId, sessions }));
      state.persistence = true;
    } catch (error) {
      state.persistence = false;
      console.warn('Condenser could not persist task history:', error);
    }
    const local = document.querySelector('.local-only');
    if (local) local.textContent = state.persistence ? 'LOCAL' : 'TEMP';
  }

  function createTask(activate) {
    const previous = currentSession();
    const now = Date.now();
    const session = {
      id: uid(),
      title: 'Untitled task',
      createdAt: now,
      updatedAt: now,
      messages: [],
      runs: [],
      config: previous ? normalizeConfig(previous.config) : defaultConfig('kadmium'),
      status: 'ready',
      lastError: null
    };
    state.sessions.push(session);
    if (activate !== false) state.currentId = session.id;
    saveSessions();
    return session;
  }

  function selectSession(id) {
    if (!state.sessions.some((session) => session.id === id)) return;
    state.currentId = id;
    saveSessions();
    setView('harness');
    renderAll();
  }

  function archiveTask() {
    const session = currentSession();
    if (!session) return;
    if (!window.confirm('Archive this task and its local run history? This does not affect provider state.')) return;
    state.sessions = state.sessions.filter((item) => item.id !== session.id);
    if (!state.sessions.length) {
      createTask(false);
    }
    state.currentId = state.sessions.slice().sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
    saveSessions();
    renderAll();
  }

  function exportTask() {
    const session = currentSession();
    if (!session) return;
    const payload = {
      format: 'condenser-agent-run-v1',
      exported_at: new Date().toISOString(),
      credential_notice: 'No API keys are stored or exported by Condenser.',
      task: session
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = (session.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'condenser-task') + '.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function setView(name) {
    document.body.dataset.view = name;
    document.querySelectorAll('.view').forEach((view) => view.classList.toggle('active', view.id === 'view-' + name));
    document.querySelectorAll('.rail-nav').forEach((button) => button.classList.toggle('active', button.dataset.view === name));
    document.body.classList.remove('sessions-open');
    if (name === 'capabilities') loadPlugins();
  }

  function timeAgo(timestamp) {
    const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
    if (seconds < 60) return 'now';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h';
    return Math.floor(seconds / 86400) + 'd';
  }

  function renderSessions() {
    $('task-count').textContent = state.sessions.length;
    const list = $('session-list');
    list.replaceChildren();
    state.sessions.slice().sort((a, b) => b.updatedAt - a.updatedAt).forEach((session) => {
      const button = element('button', 'session-card' + (session.id === state.currentId ? ' active' : ''));
      button.dataset.session = session.id;
      button.append(element('span', 'session-title', session.title));
      const meta = element('span', 'session-meta');
      meta.append(element('span', session.status === 'error' ? 'session-error' : '', descriptor(session.config.kind).label));
      meta.append(element('span', '', '· ' + timeAgo(session.updatedAt)));
      button.append(meta);
      list.append(button);
    });
  }

  function renderProviderHeader() {
    const session = currentSession();
    if (!session) return;
    const item = descriptor(session.config.kind);
    $('header-provider').textContent = item.label;
    $('header-model').textContent = session.config.model || 'model required';
    $('run-provider').textContent = item.label;
    $('run-model').textContent = session.config.model || '—';
    $('run-model').title = session.config.model || '';
  }

  function executionMap(session) {
    const map = new Map();
    session.runs.forEach((run, runIndex) => {
      (run.steps || []).forEach((step) => {
        (step.tool_executions || []).forEach((execution) => {
          map.set(execution.call_id, { execution, runIndex, stepIndex: step.index });
        });
      });
    });
    return map;
  }

  function renderMessage(message, executions) {
    if (message.role === 'user') {
      const event = element('article', 'event user-event');
      const avatar = element('div', 'event-avatar', 'YOU');
      const main = element('div', 'event-main');
      const meta = element('div', 'event-meta');
      meta.append(element('span', 'event-author', 'You'));
      main.append(meta, element('div', 'event-text', message.content));
      event.append(main, avatar);
      return [event];
    }
    const event = element('article', 'event agent-event');
    const avatar = element('div', 'event-avatar');
    const mark = element('img');
    mark.src = 'assets/Logos/kadmium-mark-dark.svg';
    mark.alt = '';
    avatar.append(mark);
    const main = element('div', 'event-main');
    const meta = element('div', 'event-meta');
    meta.append(element('span', 'event-author', 'Condenser'));
    main.append(meta);
    const toolCalls = message.tool_calls || [];
    const content = message.content || (toolCalls.length
      ? 'Requested ' + toolCalls.length + (toolCalls.length === 1 ? ' tool call.' : ' tool calls.')
      : 'No text returned.');
    main.append(element('div', 'event-text', content));
    event.append(avatar, main);
    const nodes = [event];
    toolCalls.forEach((call) => {
      const found = executions.get(call.id);
      if (found) nodes.push(renderToolExecution(found.execution, found.runIndex, found.stepIndex));
    });
    return nodes;
  }

  function renderToolExecution(execution, runIndex, stepIndex) {
    const card = element('div', 'tool-step');
    const head = element('div', 'tool-step-head');
    head.append(element('span', 'tool-state' + (execution.result.is_error ? ' error' : ''), execution.result.is_error ? '×' : '✓'));
    head.append(element('strong', '', execution.tool_name));
    head.append(element('span', '', 'R' + (runIndex + 1) + '.' + stepIndex + ' · ' + execution.plugin_id));
    head.append(element('span', 'tool-time', Number(execution.duration_ms).toFixed(1) + ' ms'));
    const result = element('div', 'tool-step-result', execution.result.content);
    card.append(head, result);
    return card;
  }

  function starterCard(title, copy, prompt) {
    const button = element('button', 'starter-card');
    button.append(element('strong', '', title), element('span', '', copy));
    button.addEventListener('click', () => {
      $('prompt').value = prompt;
      $('prompt').focus();
    });
    return button;
  }

  function renderTimeline(session) {
    const timeline = $('timeline');
    timeline.replaceChildren();
    const visible = session.messages.filter((message) => message.role === 'user' || message.role === 'assistant');
    if (!visible.length) {
      const empty = element('div', 'empty-harness');
      const mark = element('div', 'empty-agent-mark');
      const image = element('img');
      image.src = 'assets/Logos/kadmium-mark-dark.svg';
      image.alt = '';
      mark.append(image);
      empty.append(mark);
      empty.append(element('h1', '', 'What should we work on?'));
      empty.append(element('p', 'body', 'Condenser can reason with your chosen provider, call explicitly enabled tools, and keep every run inspectable.'));
      const starters = element('div', 'starter-grid');
      starters.append(
        starterCard('Design a runtime', 'Compare local and hosted execution paths.', 'Help me design an inference runtime that can move between local and hosted providers.'),
        starterCard('Debug a provider', 'Build a systematic diagnosis plan.', 'Help me diagnose a model provider connection and design a practical test plan.'),
        starterCard('Plan agent tools', 'Turn a capability into a plugin.', 'Help me design a Condenser plugin that adds a useful agent capability.')
      );
      empty.append(starters);
      timeline.append(empty);
      return;
    }
    const executions = executionMap(session);
    visible.forEach((message) => {
      renderMessage(message, executions).forEach((node) => timeline.append(node));
    });
    timeline.scrollTop = timeline.scrollHeight;
  }

  function renderRunbook(session) {
    const runbook = $('runbook');
    runbook.replaceChildren();
    const events = [];
    session.runs.forEach((run, runIndex) => {
      if (run.error) events.push({ type: 'error', runIndex, detail: run.error });
      (run.steps || []).forEach((step) => {
        events.push({ type: 'model', runIndex, step, detail: step.response.finish_reason || 'complete' });
        (step.tool_executions || []).forEach((execution) => events.push({ type: 'tool', runIndex, step, execution, detail: execution.tool_name }));
      });
    });
    $('run-steps').textContent = events.filter((event) => event.type === 'model').length;
    $('runbook-count').textContent = events.length + (events.length === 1 ? ' event' : ' events');
    if (!events.length) {
      runbook.append(element('p', 'empty-copy caption', 'Model requests and tool executions will appear here in order.'));
      return;
    }
    events.slice(-40).forEach((event, index) => {
      const row = element('div', 'runbook-event');
      row.append(element('span', 'runbook-index', index + 1));
      const body = element('div');
      let name;
      let detail;
      if (event.type === 'model') {
        name = 'Model step R' + (event.runIndex + 1) + '.' + event.step.index;
        detail = event.detail + ' · ' + event.step.tool_executions.length + ' tool calls';
      } else if (event.type === 'tool') {
        name = 'Tool · ' + event.execution.tool_name;
        detail = Number(event.execution.duration_ms).toFixed(1) + ' ms' + (event.execution.result.is_error ? ' · error' : ' · complete');
      } else {
        name = 'Turn failed';
        detail = event.detail;
      }
      body.append(element('div', 'runbook-name', name), element('div', 'runbook-detail', detail));
      row.append(body);
      runbook.append(row);
    });
  }

  function renderInspector(session) {
    renderProviderHeader();
    const totals = session.runs.reduce((sum, run) => {
      const usage = run.usage || {};
      return sum + (Number(usage.total_tokens) || (Number(usage.input_tokens) || 0) + (Number(usage.output_tokens) || 0));
    }, 0);
    $('run-tokens').textContent = totals ? totals.toLocaleString() : '—';
    $('run-tools').textContent = session.config.enable_tools ? 'Enabled · ' + pluginTools() + ' available' : 'Disabled';
    renderRunbook(session);
    if (document.activeElement !== $('system-prompt')) $('system-prompt').value = session.config.system_prompt;
  }

  function renderTask() {
    const session = currentSession();
    if (!session) return;
    const working = state.busy && state.busySessionId === session.id;
    const status = working ? 'running' : session.status;
    if (document.activeElement !== $('task-title')) $('task-title').value = session.title;
    $('task-kicker').textContent = 'TASK / ' + session.id.slice(0, 8);
    $('task-status').className = 'run-status ' + status;
    $('task-status').textContent = status === 'running' ? 'Working' : status === 'complete' ? 'Complete' : status === 'error' ? 'Needs attention' : 'Ready';
    $('enable-tools').checked = session.config.enable_tools;
    $('tool-permission-label').textContent = session.config.enable_tools ? 'Tools on · ' + pluginTools() : 'Tools off';
    if (document.activeElement !== $('step-budget')) $('step-budget').value = session.config.max_steps;
    $('composer-context').classList.toggle('hidden', !session.config.system_prompt.trim());
    $('running-indicator').classList.toggle('hidden', !working);
    $('running-detail').textContent = descriptor(session.config.kind).label + ' · up to ' + session.config.max_steps + ' steps · tools ' + (session.config.enable_tools ? 'enabled' : 'disabled');
    $('send').disabled = state.busy;
    $('send').querySelector('span').textContent = state.busy ? 'Working…' : 'Run task';
    $('provider-button').disabled = state.busy;
    $('composer-status').classList.toggle('error', status === 'error');
    $('composer-status').textContent = working ? 'The agent loop is running.' : status === 'error' ? session.lastError : status === 'complete' ? 'Run complete.' : '';
    renderTimeline(session);
    renderInspector(session);
  }

  function renderCapabilitySummary() {
    const summary = $('capability-summary');
    summary.replaceChildren();
    const tools = state.plugins.flatMap((plugin) => plugin.manifest.tools.map((tool) => plugin.manifest.id + '__' + tool.name));
    if (!tools.length) {
      summary.append(element('p', 'empty-copy caption', 'No tools discovered yet.'));
      return;
    }
    tools.slice(0, 7).forEach((name) => summary.append(element('div', 'capability-row', name)));
    if (tools.length > 7) summary.append(element('div', 'capability-row', '+ ' + (tools.length - 7) + ' more'));
  }

  function renderAll() {
    renderSessions();
    renderTask();
    renderCapabilitySummary();
  }

  function pluginTools() {
    return state.plugins.reduce((count, plugin) => count + plugin.manifest.tools.length, 0);
  }

  async function loadPlugins() {
    try {
      const inventory = await api('/api/v1/plugins');
      state.plugins = inventory.plugins || [];
      const tools = pluginTools();
      $('stat-plugins').textContent = state.plugins.length;
      $('stat-tools').textContent = tools;
      $('nav-plugin-count').textContent = state.plugins.length;
      renderPlugins(inventory.diagnostics || []);
      renderCapabilitySummary();
      const session = currentSession();
      if (session) {
        $('tool-permission-label').textContent = session.config.enable_tools ? 'Tools on · ' + tools : 'Tools off';
        $('run-tools').textContent = session.config.enable_tools ? 'Enabled · ' + tools + ' available' : 'Disabled';
      }
    } catch (error) {
      $('plugin-list').replaceChildren(element('div', 'empty-registry', error.message));
    }
  }

  function renderPlugins(diagnostics) {
    $('plugin-diagnostics').replaceChildren();
    diagnostics.forEach((diagnostic) => $('plugin-diagnostics').append(element('div', 'diagnostic', diagnostic.path + ': ' + diagnostic.message)));
    const list = $('plugin-list');
    list.replaceChildren();
    if (!state.plugins.length) {
      const empty = element('div', 'empty-registry');
      empty.append(element('h3', '', 'No capabilities registered yet.'));
      empty.append(element('p', '', 'Scaffold a plugin in ./plugins or ~/.condenser/plugins, then refresh this registry. New manifests are discovered without restarting Condenser.'));
      empty.append(element('code', '', 'cargo run -p kadmium-condenser -- plugin scaffold my-tool'));
      list.append(empty);
      return;
    }
    state.plugins.forEach((plugin) => {
      const card = element('article', 'plugin-card');
      const head = element('div', 'plugin-head');
      const identity = element('div', 'plugin-identity');
      identity.append(element('h3', '', plugin.manifest.name), element('p', '', plugin.manifest.description || 'No description supplied.'));
      head.append(identity, element('span', 'plugin-version', 'v' + plugin.manifest.version));
      card.append(head, element('div', 'plugin-root', plugin.manifest.id + ' · ' + plugin.root));
      const tools = element('div', 'plugin-tools');
      plugin.manifest.tools.forEach((tool) => {
        const row = element('div', 'tool-row');
        const info = element('div');
        info.append(element('div', 'tool-name', plugin.manifest.id + '__' + tool.name));
        info.append(element('div', 'tool-description', tool.description));
        info.append(element('div', 'tool-schema', JSON.stringify(tool.input_schema)));
        const invoke = element('button', 'secondary-btn', 'Test tool');
        invoke.dataset.plugin = plugin.manifest.id;
        invoke.dataset.tool = tool.name;
        row.append(info, invoke);
        tools.append(row);
      });
      if (!plugin.manifest.tools.length) tools.append(element('p', 'empty-copy caption', 'This plugin declares no tools.'));
      card.append(tools);
      list.append(card);
    });
  }

  function sampleArguments(schema) {
    if (!schema || schema.type !== 'object') return {};
    if (schema.const !== undefined) return schema.const;
    const output = {};
    Object.entries(schema.properties || {}).forEach(([name, definition]) => {
      if (definition.const !== undefined) output[name] = definition.const;
      else if (Array.isArray(definition.enum) && definition.enum.length) output[name] = definition.enum[0];
      else if (definition.type === 'string') output[name] = name === 'text' ? 'Hello from Condenser' : 'example';
      else if (definition.type === 'number' || definition.type === 'integer') output[name] = 1;
      else if (definition.type === 'boolean') output[name] = true;
      else if (definition.type === 'array') output[name] = [];
      else if (definition.type === 'object') output[name] = sampleArguments(definition);
    });
    return output;
  }

  function openToolConsole(pluginId, toolName) {
    const plugin = state.plugins.find((item) => item.manifest.id === pluginId);
    const tool = plugin && plugin.manifest.tools.find((item) => item.name === toolName);
    if (!plugin || !tool) return;
    state.selectedTool = { pluginId, toolName };
    $('tool-dialog-title').textContent = plugin.manifest.id + '__' + tool.name;
    $('tool-dialog-description').textContent = tool.description;
    $('tool-arguments').value = JSON.stringify(sampleArguments(tool.input_schema), null, 2);
    $('tool-result').classList.add('hidden');
    $('tool-layer').classList.remove('hidden');
    $('tool-arguments').focus();
  }

  async function invokeSelectedTool() {
    if (!state.selectedTool) return;
    let argumentsValue;
    try {
      argumentsValue = JSON.parse($('tool-arguments').value);
    } catch (error) {
      showToolResult('Invalid JSON: ' + error.message, true);
      return;
    }
    $('invoke-tool').disabled = true;
    $('invoke-tool').textContent = 'Running…';
    try {
      const result = await api('/api/v1/plugins/invoke', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ plugin_id: state.selectedTool.pluginId, tool: state.selectedTool.toolName, arguments: argumentsValue })
      });
      showToolResult(result.content, result.is_error);
    } catch (error) {
      showToolResult(error.message, true);
    } finally {
      $('invoke-tool').disabled = false;
      $('invoke-tool').textContent = 'Invoke tool';
    }
  }

  function showToolResult(text, isError) {
    $('tool-result').textContent = text;
    $('tool-result').classList.toggle('error', Boolean(isError));
    $('tool-result').classList.remove('hidden');
  }

  function closeLayer(name) {
    $(name + '-layer').classList.add('hidden');
  }

  function applyProviderDefaults(kind) {
    const item = descriptor(kind);
    $('provider').value = kind;
    $('base-url').value = item.default_base_url;
    $('model').value = item.default_model;
    $('model').placeholder = item.default_model || 'provider-model-id';
    $('api-key').value = state.credentials[kind] || '';
    $('api-key').placeholder = item.requires_api_key ? 'required' : 'optional for local providers';
    $('key-hint').textContent = 'Memory-only. CLI alternative: ' + item.default_api_key_env + '.';
    $('provider-note').textContent = item.note;
    $('provider-note').classList.toggle('warn', kind === 'kadmium');
    $('model-options').replaceChildren();
    $('models-status').textContent = '';
  }

  function openSettings() {
    if (state.busy) return;
    const session = currentSession();
    if (!session) return;
    applyProviderDefaults(session.config.kind);
    $('base-url').value = session.config.base_url;
    $('model').value = session.config.model;
    $('temperature').value = session.config.sampling.temperature === undefined ? '' : session.config.sampling.temperature;
    $('max-tokens').value = session.config.sampling.max_tokens === undefined ? '' : session.config.sampling.max_tokens;
    $('settings-layer').classList.remove('hidden');
  }

  function saveSettings() {
    const session = currentSession();
    if (!session) return;
    const kind = $('provider').value;
    const config = session.config;
    config.kind = kind;
    config.base_url = $('base-url').value.trim();
    config.model = $('model').value.trim();
    config.sampling = {};
    if ($('temperature').value !== '') config.sampling.temperature = Number($('temperature').value);
    if ($('max-tokens').value !== '') config.sampling.max_tokens = Number($('max-tokens').value);
    if ($('api-key').value) state.credentials[kind] = $('api-key').value;
    else delete state.credentials[kind];
    session.updatedAt = Date.now();
    session.lastError = null;
    session.status = 'ready';
    saveSessions();
    closeLayer('settings');
    renderAll();
  }

  function modalProviderConfig() {
    return {
      kind: $('provider').value,
      model: $('model').value.trim(),
      base_url: $('base-url').value.trim(),
      api_key: $('api-key').value || null,
      timeout_secs: 300
    };
  }

  async function loadModels() {
    $('load-models').disabled = true;
    $('models-status').classList.remove('error');
    $('models-status').textContent = 'loading…';
    try {
      const models = await api('/api/v1/models', {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ provider: modalProviderConfig() })
      });
      $('model-options').replaceChildren();
      models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.display_name || model.id;
        $('model-options').append(option);
      });
      if (!$('model').value && models.length) $('model').value = models[0].id;
      $('models-status').textContent = models.length + ' found';
    } catch (error) {
      $('models-status').textContent = error.message;
      $('models-status').classList.add('error');
    } finally {
      $('load-models').disabled = false;
    }
  }

  async function loadSystem() {
    try {
      const node = await api('/api/v1/system');
      $('system-badge').classList.add('ok');
      $('system-badge').querySelector('span:last-child').textContent = node.hostname + ' · ' + node.backend + ' · ' + Number(node.fast_memory_gb).toFixed(1) + ' GB fast';
    } catch (error) {
      $('system-badge').classList.add('warn');
      $('system-badge').querySelector('span:last-child').textContent = 'Host probe unavailable';
    }
  }

  function updateTaskConfig() {
    const session = currentSession();
    if (!session) return;
    session.updatedAt = Date.now();
    saveSessions();
    renderSessions();
    renderInspector(session);
  }

  async function send() {
    if (state.busy) return;
    const session = currentSession();
    const prompt = $('prompt').value.trim();
    if (!session || !prompt) return;
    if (!session.config.model.trim()) {
      session.status = 'error';
      session.lastError = 'Choose a model in provider settings before running this task.';
      renderTask();
      openSettings();
      return;
    }
    const sessionId = session.id;
    const messages = session.messages.filter((message) => message.role !== 'system');
    messages.push({ role: 'user', content: prompt, tool_calls: [], is_error: false });
    if (session.config.system_prompt.trim()) {
      messages.unshift({ role: 'system', content: session.config.system_prompt.trim(), tool_calls: [], is_error: false });
    }
    const provider = {
      kind: session.config.kind,
      model: session.config.model,
      base_url: session.config.base_url,
      api_key: state.credentials[session.config.kind] || null,
      timeout_secs: 300
    };
    state.busy = true;
    state.busySessionId = sessionId;
    session.status = 'running';
    session.lastError = null;
    saveSessions();
    renderAll();
    try {
      const report = await api('/api/v1/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          provider,
          messages,
          enable_tools: session.config.enable_tools,
          max_steps: session.config.max_steps,
          sampling: session.config.sampling
        })
      });
      const target = state.sessions.find((item) => item.id === sessionId);
      if (target) {
        target.messages = report.messages.filter((message) => message.role !== 'system');
        target.runs.push({ id: uid(), at: Date.now(), steps: report.steps, usage: report.usage, diagnostics: report.diagnostics });
        target.status = 'complete';
        target.lastError = null;
        target.updatedAt = Date.now();
        if (target.title === 'Untitled task') target.title = prompt.length > 72 ? prompt.slice(0, 69) + '…' : prompt;
      }
      if (state.currentId === sessionId) $('prompt').value = '';
    } catch (error) {
      const target = state.sessions.find((item) => item.id === sessionId);
      if (target) {
        target.status = 'error';
        target.lastError = error.message;
        target.updatedAt = Date.now();
        target.runs.push({ id: uid(), at: Date.now(), steps: [], usage: {}, error: error.message });
      }
    } finally {
      state.busy = false;
      state.busySessionId = null;
      saveSessions();
      renderAll();
      $('prompt').focus();
    }
  }

  function bindEvents() {
    document.querySelectorAll('.rail-nav').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
    $('session-list').addEventListener('click', (event) => {
      const card = event.target.closest('[data-session]');
      if (card) selectSession(card.dataset.session);
    });
    $('new-task').addEventListener('click', () => {
      const session = createTask(true);
      selectSession(session.id);
      $('prompt').focus();
    });
    $('archive-task').addEventListener('click', archiveTask);
    $('export-task').addEventListener('click', exportTask);
    $('task-title').addEventListener('input', () => {
      const session = currentSession();
      if (!session) return;
      session.title = $('task-title').value;
      session.updatedAt = Date.now();
      renderSessions();
      saveSessions();
    });
    $('task-title').addEventListener('blur', () => {
      const session = currentSession();
      if (!session) return;
      if (!session.title.trim()) session.title = 'Untitled task';
      $('task-title').value = session.title;
      saveSessions();
      renderSessions();
    });
    $('send').addEventListener('click', send);
    $('prompt').addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') send();
    });
    $('enable-tools').addEventListener('change', () => {
      const session = currentSession();
      if (!session) return;
      session.config.enable_tools = $('enable-tools').checked;
      session.updatedAt = Date.now();
      saveSessions();
      renderTask();
      renderSessions();
    });
    $('step-budget').addEventListener('change', () => {
      const session = currentSession();
      if (!session) return;
      session.config.max_steps = Math.max(1, Math.min(32, Number($('step-budget').value) || 8));
      $('step-budget').value = session.config.max_steps;
      updateTaskConfig();
    });
    $('save-instructions').addEventListener('click', () => {
      const session = currentSession();
      if (!session) return;
      session.config.system_prompt = $('system-prompt').value;
      session.updatedAt = Date.now();
      saveSessions();
      renderTask();
      renderSessions();
      $('save-instructions').textContent = 'Applied';
      setTimeout(() => { $('save-instructions').textContent = 'Apply'; }, 1200);
    });
    $('edit-instructions').addEventListener('click', () => {
      document.body.classList.remove('context-closed');
      $('system-prompt').focus();
    });
    $('provider-button').addEventListener('click', openSettings);
    $('provider').addEventListener('change', () => applyProviderDefaults($('provider').value));
    $('save-settings').addEventListener('click', saveSettings);
    $('cancel-settings').addEventListener('click', () => closeLayer('settings'));
    $('load-models').addEventListener('click', loadModels);
    $('refresh-plugins').addEventListener('click', loadPlugins);
    $('open-capabilities').addEventListener('click', () => setView('capabilities'));
    $('plugin-list').addEventListener('click', (event) => {
      const button = event.target.closest('[data-plugin][data-tool]');
      if (button) openToolConsole(button.dataset.plugin, button.dataset.tool);
    });
    $('invoke-tool').addEventListener('click', invokeSelectedTool);
    document.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => closeLayer(button.dataset.close)));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closeLayer('settings');
        closeLayer('tool');
        document.body.classList.remove('sessions-open');
      }
    });
    $('toggle-context').addEventListener('click', () => document.body.classList.toggle('context-closed'));
    $('close-context').addEventListener('click', () => document.body.classList.add('context-closed'));
    $('toggle-sessions').addEventListener('click', () => document.body.classList.toggle('sessions-open'));
  }

  async function loadProviders() {
    const body = await api('/api/v1/providers');
    state.providers = body.providers || [];
    $('provider').replaceChildren();
    state.providers.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.kind;
      option.textContent = item.label;
      $('provider').append(option);
    });
  }

  async function init() {
    bindEvents();
    if (window.matchMedia('(max-width: 980px)').matches) document.body.classList.add('context-closed');
    await loadProviders();
    loadSessions();
    renderAll();
    await Promise.all([loadPlugins(), loadSystem()]);
  }

  init().catch((error) => {
    console.error(error);
    $('composer-status').textContent = error.message;
    $('composer-status').classList.add('error');
  });
})();
