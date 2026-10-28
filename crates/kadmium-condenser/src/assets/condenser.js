(function () {
  'use strict';

  const STORAGE_KEY = 'condenser.agent.tasks.v1';
  const FALLBACK_PROVIDERS = {
    'mcl4-coe': { label: 'MCL4-COE (DeepSeek-V4.1)', base: 'http://127.0.0.1:7861', model: 'dsv41-mcl4-step_10000', keyEnv: 'MCL4_API_KEY', note: 'DeepSeek-V4.1-Flash-1017-NVFP4 (40L, 384+1 MoE, MLA 64Hx512, Engram [1,14], Dual-P4608 + MCL4DS41 step_10000) + 9-operator MCL4-COE engine.' },
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
    persistence: true,
    contextPanel: 'run',
    contextWide: false,
    workspace: { root: '', name: 'Workspace', separator: '/' },
    files: {
      path: '', parent: null, selected: null, loaded: false, request: 0, previewRequest: 0,
      editMode: false, showDiff: false, originalContent: '', dirty: false, saving: false,
      version: 0, saveTimer: null, savePromise: null, operation: false
    },
    terminal: { cwd: '', running: false, history: [], historyIndex: 0 },
    browser: { current: '', history: [], index: -1 },
    activity: null,
    reveal: null,
    lastElapsed: 0
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
    const item = descriptor(kind || 'mcl4-coe');
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
    let kind = raw && FALLBACK_PROVIDERS[raw.kind] ? raw.kind : 'mcl4-coe';
    const isLegacyKadmiumDefault =
      kind === 'kadmium' &&
      (!raw.base_url || raw.base_url === 'http://127.0.0.1:7848') &&
      (!raw.model || raw.model === 'deepseek-v3-671b-moe-q4k');
    if (isLegacyKadmiumDefault) {
      kind = 'mcl4-coe';
      raw = Object.assign({}, raw, {
        kind: 'mcl4-coe',
        base_url: 'http://127.0.0.1:7861',
        model: 'dsv41-mcl4-step_10000'
      });
    }
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
      config: previous ? normalizeConfig(previous.config) : defaultConfig('mcl4-coe'),
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
    if (state.reveal && state.reveal.sessionId === session.id) finishTokenReveal(state.reveal);
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

  function saveDockLayout() {
    try {
      localStorage.setItem('condenser.workspace.dock.v1', JSON.stringify({ panel: state.contextPanel, wide: state.contextWide }));
    } catch (_) { /* The dock remains usable without persistence. */ }
  }

  function loadDockLayout() {
    try {
      const saved = JSON.parse(localStorage.getItem('condenser.workspace.dock.v1') || 'null');
      if (saved && ['run', 'files', 'terminal', 'browser'].includes(saved.panel)) state.contextPanel = saved.panel;
      if (saved) state.contextWide = Boolean(saved.wide);
    } catch (_) { /* Ignore invalid or unavailable layout storage. */ }
    document.body.classList.toggle('context-wide', state.contextWide);
  }

  function setContextPanel(name, reveal) {
    if (!['run', 'files', 'terminal', 'browser'].includes(name)) return;
    state.contextPanel = name;
    document.body.dataset.contextPanel = name;
    if (reveal !== false) document.body.classList.remove('context-closed');
    document.querySelectorAll('[data-context-tab]').forEach((button) => {
      const active = button.dataset.contextTab === name;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('.context-panel').forEach((panel) => { panel.hidden = panel.id !== 'context-' + name; });
    if (name === 'files' && !state.files.loaded) loadWorkspace();
    if (name === 'terminal') setTimeout(() => $('terminal-command').focus(), 0);
    saveDockLayout();
  }

  function postJson(path, payload) {
    return api(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
  }

  async function loadWorkspace() {
    try {
      state.workspace = await api('/api/v1/workspace');
      $('workspace-name').textContent = state.workspace.name || 'Workspace';
      $('workspace-name').title = state.workspace.root || '';
      await loadFiles('');
    } catch (error) {
      $('file-list').replaceChildren(element('p', 'panel-empty error', error.message));
    }
  }

  async function loadFiles(path) {
    const request = ++state.files.request;
    try {
      const listing = await postJson('/api/v1/files/list', { path: path || '' });
      if (request !== state.files.request) return;
      state.files.path = listing.path;
      state.files.parent = listing.parent;
      state.files.loaded = true;
      $('files-path').textContent = '/' + (listing.path || '');
      $('files-up').disabled = listing.parent === null;
      renderFileList(listing.entries, listing.truncated);
    } catch (error) {
      if (request === state.files.request) $('file-list').replaceChildren(element('p', 'panel-empty error', error.message));
    }
  }

  function formatBytes(size) {
    const value = Number(size) || 0;
    if (value < 1024) return value + ' B';
    if (value < 1024 * 1024) return (value / 1024).toFixed(value < 10240 ? 1 : 0) + ' KB';
    return (value / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function fileIcon(directory) {
    const span = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    span.setAttribute('viewBox', '0 0 24 24');
    span.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', directory ? 'M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z' : 'M7 3h7l4 4v14H7V3ZM14 3v5h4');
    span.append(path);
    return span;
  }

  function renderFileList(entries, truncated) {
    const list = $('file-list');
    list.replaceChildren();
    if (!entries.length) list.append(element('p', 'panel-empty', 'This folder is empty.'));
    entries.forEach((entry) => {
      const button = element('button', 'file-row ' + entry.kind + (entry.path === state.files.selected ? ' active' : ''));
      button.append(fileIcon(entry.kind === 'directory'));
      button.append(element('span', 'file-row-name', entry.name));
      if (entry.kind === 'file') button.append(element('span', 'file-row-size', formatBytes(entry.size)));
      button.title = entry.path;
      button.addEventListener('click', async () => {
        if (entry.kind === 'directory') {
          await closeFilePreview();
          if (!state.files.selected) loadFiles(entry.path);
        } else openFile(entry.path);
      });
      list.append(button);
    });
    if (truncated) list.append(element('p', 'panel-empty', 'Showing the first 500 entries.'));
  }

  function setFileStatus(text, kind) {
    $('file-save-status').textContent = text;
    $('file-save-status').className = 'file-save-status' + (kind ? ' ' + kind : '');
  }

  function updateFileViews() {
    const files = state.files;
    $('file-preview-content').classList.toggle('hidden', files.editMode || files.showDiff);
    $('file-editor').classList.toggle('hidden', !files.editMode || files.showDiff);
    $('file-diff').classList.toggle('hidden', !files.showDiff);
    $('edit-file').textContent = files.editMode ? 'Read' : 'Edit';
    $('edit-file').setAttribute('aria-pressed', String(files.editMode));
    $('diff-file').setAttribute('aria-pressed', String(files.showDiff));
    $('save-file').disabled = !files.dirty || files.saving;
  }

  function resetFileEditor() {
    const files = state.files;
    clearTimeout(files.saveTimer);
    files.editMode = false;
    files.showDiff = false;
    files.originalContent = '';
    files.dirty = false;
    files.saving = false;
    files.version = 0;
    $('edit-file').disabled = false;
    $('file-editor').value = '';
    $('file-diff').replaceChildren();
    $('rename-file-form').classList.add('hidden');
    setFileStatus('Autosave on');
    updateFileViews();
  }

  async function writeCurrentFile() {
    const files = state.files;
    if (!files.selected || !files.dirty || files.saving) return;
    clearTimeout(files.saveTimer);
    const path = files.selected;
    const version = files.version;
    const content = $('file-editor').value;
    files.saving = true;
    setFileStatus('Saving…', 'saving');
    updateFileViews();
    try {
      const result = await postJson('/api/v1/files/write', { path, content });
      if (files.selected !== path) return;
      $('preview-file-meta').textContent = formatBytes(result.size) + ' · ' + result.path;
      if (files.version === version) {
        files.dirty = false;
        setFileStatus('Saved');
      } else {
        setFileStatus('Pending autosave');
        queueFileSave();
      }
      if (!files.editMode) $('file-preview-content').textContent = content;
    } catch (error) {
      if (files.selected === path) setFileStatus(error.message, 'error');
    } finally {
      if (files.selected === path) {
        files.saving = false;
        updateFileViews();
        if (files.showDiff) renderFileDiff();
      }
    }
  }

  function saveFile() {
    const files = state.files;
    if (!files.selected || !files.dirty) return Promise.resolve();
    if (files.savePromise) return files.savePromise;
    const operation = writeCurrentFile();
    files.savePromise = operation;
    return operation.finally(() => {
      if (files.savePromise === operation) files.savePromise = null;
    });
  }

  function queueFileSave() {
    const files = state.files;
    clearTimeout(files.saveTimer);
    files.saveTimer = setTimeout(saveFile, 800);
  }

  async function flushFileEdit() {
    const files = state.files;
    if (files.savePromise) await files.savePromise;
    if (files.dirty) await saveFile();
    return !files.dirty;
  }

  function setFileEditMode(enabled) {
    const files = state.files;
    if (enabled === files.editMode) return;
    if (!enabled) {
      flushFileEdit();
      $('file-preview-content').textContent = $('file-editor').value;
    } else {
      $('file-editor').value = $('file-preview-content').textContent;
    }
    files.editMode = enabled;
    files.showDiff = false;
    updateFileViews();
    if (enabled) $('file-editor').focus();
  }

  function appendDiffLine(container, kind, number, sign, content) {
    const row = element('div', 'diff-line ' + kind);
    row.append(element('span', 'diff-line-number', number || ''));
    row.append(element('span', 'diff-line-sign', sign));
    row.append(element('span', 'diff-line-code', content));
    container.append(row);
  }

  function renderFileDiff() {
    const before = state.files.originalContent.split('\n');
    const current = state.files.editMode ? $('file-editor').value : $('file-preview-content').textContent;
    const after = current.split('\n');
    let prefix = 0;
    while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix += 1;
    let suffix = 0;
    while (suffix < before.length - prefix && suffix < after.length - prefix && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix += 1;
    const removed = before.slice(prefix, before.length - suffix);
    const added = after.slice(prefix, after.length - suffix);
    const container = $('file-diff');
    container.replaceChildren();
    const summary = element('div', 'diff-summary');
    summary.append(element('span', 'removed', '−' + removed.length + ' removed'));
    summary.append(document.createTextNode(' · '));
    summary.append(element('span', 'added', '+' + added.length + ' added'));
    container.append(summary);
    if (!removed.length && !added.length) {
      container.append(element('p', 'panel-empty', 'No changes from the opened version.'));
      return;
    }
    before.slice(Math.max(0, prefix - 3), prefix).forEach((line, index, lines) => appendDiffLine(container, 'context', prefix - lines.length + index + 1, ' ', line));
    removed.slice(0, 400).forEach((line, index) => appendDiffLine(container, 'removed', prefix + index + 1, '−', line));
    added.slice(0, 400).forEach((line, index) => appendDiffLine(container, 'added', prefix + index + 1, '+', line));
    if (removed.length > 400 || added.length > 400) container.append(element('div', 'diff-gap', 'Large change block truncated in this view'));
    after.slice(after.length - suffix, after.length - suffix + 3).forEach((line, index) => appendDiffLine(container, 'context', after.length - suffix + index + 1, ' ', line));
  }

  function setFileOperationStatus(text, isError) {
    $('file-operation-status').textContent = text || '';
    $('file-operation-status').classList.toggle('error', Boolean(isError));
    $('file-operation-status').classList.toggle('hidden', !text);
  }

  function showNewFileForm() {
    setFileOperationStatus('');
    $('rename-file-form').classList.add('hidden');
    const prefix = state.files.path ? state.files.path + '/' : '';
    $('new-file-path').value = prefix + 'untitled.txt';
    $('new-file-form').classList.remove('hidden');
    $('new-file-path').focus();
    $('new-file-path').setSelectionRange(prefix.length, prefix.length + 8);
  }

  async function createFile() {
    const files = state.files;
    const path = $('new-file-path').value.trim();
    if (!path || files.operation) return;
    files.operation = true;
    try {
      const result = await postJson('/api/v1/files/create', { path, content: '' });
      $('new-file-form').classList.add('hidden');
      await loadFiles(files.path);
      await openFile(result.path);
      setFileEditMode(true);
      setFileOperationStatus('Created ' + result.path);
    } catch (error) {
      setFileOperationStatus(error.message, true);
    } finally {
      files.operation = false;
    }
  }

  function showRenameFileForm() {
    if (!state.files.selected) return;
    setFileOperationStatus('');
    $('new-file-form').classList.add('hidden');
    $('rename-file-path').value = state.files.selected;
    $('rename-file-form').classList.remove('hidden');
    $('rename-file-path').focus();
    $('rename-file-path').select();
  }

  async function renameSelectedFile() {
    const files = state.files;
    const newPath = $('rename-file-path').value.trim();
    const path = files.selected;
    if (!path || !newPath || files.operation) return;
    if (newPath === path) {
      $('rename-file-form').classList.add('hidden');
      return;
    }
    if (!await flushFileEdit()) return;
    files.operation = true;
    try {
      const result = await postJson('/api/v1/files/rename', { path, new_path: newPath });
      $('rename-file-form').classList.add('hidden');
      await loadFiles(files.path);
      await openFile(result.path);
      setFileOperationStatus('Renamed to ' + result.path);
    } catch (error) {
      setFileOperationStatus(error.message, true);
    } finally {
      files.operation = false;
    }
  }

  async function deleteSelectedFile() {
    const files = state.files;
    const path = files.selected;
    if (!path || files.operation) return;
    if (!await flushFileEdit()) return;
    files.operation = true;
    try {
      const result = await postJson('/api/v1/files/delete', { path });
      await closeFilePreview();
      await loadFiles(files.path);
      setFileOperationStatus('Moved to ' + result.trash_path);
    } catch (error) {
      setFileOperationStatus(error.message, true);
    } finally {
      files.operation = false;
    }
  }

  async function closeFilePreview() {
    if (!await flushFileEdit()) return;
    resetFileEditor();
    state.files.previewRequest += 1;
    state.files.selected = null;
    $('context-files').classList.remove('has-preview');
    $('file-preview').hidden = true;
    document.querySelectorAll('.file-row.active').forEach((row) => row.classList.remove('active'));
  }

  async function openFile(path) {
    if (!await flushFileEdit()) return;
    resetFileEditor();
    const request = ++state.files.previewRequest;
    state.files.selected = path;
    document.querySelectorAll('.file-row').forEach((row) => row.classList.toggle('active', row.title === path));
    $('context-files').classList.add('has-preview');
    $('file-preview').hidden = false;
    $('preview-file-name').textContent = path.split('/').pop();
    $('preview-file-meta').textContent = path;
    $('file-preview-content').textContent = 'Loading…';
    try {
      const file = await postJson('/api/v1/files/read', { path });
      if (request !== state.files.previewRequest) return;
      state.files.originalContent = file.content;
      $('preview-file-meta').textContent = formatBytes(file.size) + (file.truncated ? ' · preview truncated at 1 MiB' : ' · ' + file.path);
      $('file-preview-content').textContent = file.content;
      $('file-editor').value = file.content;
      $('edit-file').disabled = file.truncated;
      setFileStatus(file.truncated ? 'Preview only · 1 MiB limit' : 'Autosave on');
      $('file-preview-content').scrollTop = 0;
    } catch (error) {
      if (request === state.files.previewRequest) {
        $('file-preview-content').textContent = error.message;
        $('edit-file').disabled = true;
      }
    }
  }

  function terminalPath() {
    return '/' + (state.terminal.cwd || '');
  }

  function appendTerminalOutput(command, cwd, result) {
    const entry = element('div', 'terminal-entry');
    entry.append(element('div', 'terminal-command-line', '/' + (cwd || '') + ' › ' + command));
    if (result) {
      if (result.stdout) entry.append(element('pre', 'terminal-stream', result.stdout));
      if (result.stderr) entry.append(element('pre', 'terminal-stream stderr', result.stderr));
      const failed = result.timed_out || result.exit_code !== 0;
      let detail = result.timed_out ? 'Stopped after 20 s' : 'Exit ' + result.exit_code + ' · ' + result.duration_ms + ' ms';
      if (result.stdout_truncated || result.stderr_truncated) detail += ' · output truncated at 128 KiB';
      entry.append(element('div', 'terminal-meta' + (failed ? ' error' : ''), detail));
    }
    $('terminal-output').append(entry);
    $('terminal-output').scrollTop = $('terminal-output').scrollHeight;
  }

  async function executeTerminal() {
    if (state.terminal.running) return;
    const command = $('terminal-command').value.trim();
    if (!command) return;
    const cwd = state.terminal.cwd;
    state.terminal.history.push(command);
    state.terminal.historyIndex = state.terminal.history.length;
    state.terminal.running = true;
    $('terminal-run').disabled = true;
    $('terminal-command').value = '';
    $('terminal-status').className = 'terminal-status running';
    $('terminal-status').textContent = 'Running…';
    try {
      const result = await postJson('/api/v1/terminal/execute', { command, cwd });
      state.terminal.cwd = result.cwd || '';
      $('terminal-cwd').textContent = terminalPath();
      $('terminal-cwd').title = state.workspace.root ? state.workspace.root + state.workspace.separator + state.terminal.cwd.replaceAll('/', state.workspace.separator) : terminalPath();
      appendTerminalOutput(command, cwd, result);
      $('terminal-status').className = 'terminal-status' + (result.timed_out || result.exit_code !== 0 ? ' error' : '');
      $('terminal-status').textContent = result.timed_out ? 'Command timed out' : 'Finished with exit code ' + result.exit_code;
    } catch (error) {
      appendTerminalOutput(command, cwd, { stdout: '', stderr: error.message, exit_code: null, timed_out: false, duration_ms: 0 });
      $('terminal-status').className = 'terminal-status error';
      $('terminal-status').textContent = error.message;
    } finally {
      state.terminal.running = false;
      $('terminal-run').disabled = false;
      $('terminal-command').focus();
    }
  }

  function normalizeBrowserUrl(raw) {
    const value = raw.trim();
    if (!value) return null;
    const explicitScheme = /^[a-zA-Z][a-zA-Z\d+.-]*:/.test(value) && !/^[^/]+:\d+(?:\/|$)/.test(value);
    const localAddress = /^(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$)/i.test(value);
    const candidate = explicitScheme ? value : (localAddress ? 'http://' : 'https://') + value;
    const url = new URL(candidate);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Enter an HTTP(S) address without embedded credentials.');
    return url.href;
  }

  function updateBrowserHistory() {
    $('browser-back').disabled = !state.browser.current;
    $('browser-forward').disabled = !state.browser.current;
    $('browser-reload').disabled = !state.browser.current;
    $('browser-external').disabled = !state.browser.current;
  }

  function navigateBrowser(raw, remember) {
    let url;
    try {
      url = normalizeBrowserUrl(raw);
    } catch (error) {
      $('browser-status').classList.remove('hidden');
      $('browser-status').textContent = error.message;
      return;
    }
    if (!url) return;
    state.browser.current = url;
    if (remember !== false) {
      state.browser.history = state.browser.history.slice(0, state.browser.index + 1);
      state.browser.history.push(url);
      if (state.browser.history.length > 40) state.browser.history.shift();
      state.browser.index = state.browser.history.length - 1;
    }
    $('browser-url').value = url;
    $('browser-home').classList.add('hidden');
    $('browser-frame').classList.remove('hidden');
    $('browser-status').classList.remove('hidden');
    $('browser-status').textContent = 'Loading ' + url;
    $('browser-frame').src = url;
    updateBrowserHistory();
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
    const query = ($('session-search').value || '').trim().toLowerCase();
    const sessions = state.sessions
      .filter((session) => !query || (session.title + ' ' + descriptor(session.config.kind).label).toLowerCase().includes(query))
      .sort((a, b) => b.updatedAt - a.updatedAt);
    if (!sessions.length) {
      list.append(element('p', 'session-search-empty', query ? 'No tasks match your search.' : 'Your next task will appear here.'));
      return;
    }
    sessions.forEach((session) => {
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
    mark.src = 'assets/condenser-mark.svg';
    mark.alt = '';
    avatar.append(mark);
    const main = element('div', 'event-main');
    const meta = element('div', 'event-meta');
    meta.append(element('span', 'event-author', 'Blades'));
    main.append(meta);
    if (message.reasoning && String(message.reasoning).trim()) {
      const trace = element('details', 'reasoning-trace');
      trace.append(
        element('summary', '', 'Reasoning trace (#T$# / <think>)'),
        element('div', 'reasoning-body', String(message.reasoning).trim())
      );
      main.append(trace);
    }
    const toolCalls = message.tool_calls || [];
    const revealing = state.reveal && state.reveal.message === message;
    const content = revealing ? state.reveal.visibleText : message.content || (toolCalls.length
      ? 'Requested ' + toolCalls.length + (toolCalls.length === 1 ? ' tool call.' : ' tool calls.')
      : 'No text returned.');
    const text = element('div', 'event-text' + (revealing ? ' streaming-text' : ''), content);
    if (revealing) state.reveal.node = text;
    main.append(text);
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
    timeline.closest('.agent-stage').classList.toggle('is-empty', !visible.length);
    if (!visible.length) {
      const empty = element('div', 'empty-harness');
      const mark = element('div', 'empty-agent-mark');
      const image = element('img');
      image.src = 'assets/condenser-mark.svg';
      image.alt = '';
      mark.append(image);
      empty.append(mark);
      empty.append(element('p', 'empty-kicker label', 'BLADES · YOUR EVERYDAY AGENT'));
      empty.append(element('h1', '', 'What will we get done?'));
      empty.append(element('p', 'body', 'One focused workspace for code, research, computer workflows, and everything in between.'));
      const starters = element('div', 'starter-grid');
      starters.append(
        starterCard('Build & debug', 'Work through a coding problem.', 'Help me break down and solve a coding problem with a clear, practical implementation plan.'),
        starterCard('Research', 'Compare ideas and find a direction.', 'Help me research a topic, compare the important tradeoffs, and organize what I learn.'),
        starterCard('Computer workflows', 'Plan a repeatable desktop task.', 'Help me design a computer-use workflow with clear steps, checkpoints, and safe tool permissions.'),
        starterCard('Everyday tasks', 'Turn a loose idea into next steps.', 'Help me organize a task, clarify the outcome, and turn it into a practical set of next steps.')
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
    const revealing = working && state.reveal && state.reveal.sessionId === session.id;
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
    $('composer-status').textContent = working ? (revealing ? 'Receiving response…' : 'The agent loop is running.') : status === 'error' ? session.lastError : status === 'complete' ? 'Run complete' + (session.lastElapsed ? ' · ' + formatElapsed(session.lastElapsed) : '') + '.' : '';
    updateActivity();
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
    const dsv41Details = $('dsv41-mcl4-settings');
    if (dsv41Details) {
      dsv41Details.open = kind === 'mcl4-coe';
    }
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
    const mcl4 = session.config.sampling.mcl4_coe || {};
    if ($('mcl4-mode')) $('mcl4-mode').value = mcl4.mode || 'uta';
    if ($('mcl4-effort')) $('mcl4-effort').value = mcl4.reasoning_effort || 'high';
    if ($('mcl4-top-k')) $('mcl4-top-k').value = mcl4.top_k === undefined || mcl4.top_k === null ? '' : String(mcl4.top_k);
    if ($('mcl4-recurrence')) $('mcl4-recurrence').value = mcl4.recurrence_steps === undefined ? '' : mcl4.recurrence_steps;
    if ($('mcl4-tabu')) $('mcl4-tabu').value = mcl4.recurrent_coverage_penalty === undefined ? '' : mcl4.recurrent_coverage_penalty;
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
    if (kind === 'mcl4-coe' || (config.model && config.model.toLowerCase().includes('v4.1'))) {
      const mcl4 = {
        mode: $('mcl4-mode') ? $('mcl4-mode').value : 'uta',
        reasoning_effort: $('mcl4-effort') ? $('mcl4-effort').value : 'high',
        adaptive_recurrent_depth: false,
        use_windowless_pager: false,
        thought_prefix: '',
        answer_prefix: ''
      };
      if ($('mcl4-top-k') && $('mcl4-top-k').value !== '') mcl4.top_k = Number($('mcl4-top-k').value);
      if ($('mcl4-recurrence') && $('mcl4-recurrence').value !== '') mcl4.recurrence_steps = Number($('mcl4-recurrence').value);
      if ($('mcl4-tabu') && $('mcl4-tabu').value !== '') mcl4.recurrent_coverage_penalty = Number($('mcl4-tabu').value);
      config.sampling.mcl4_coe = mcl4;
    }
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

  function formatElapsed(milliseconds) {
    const seconds = Math.max(0, Math.floor(milliseconds / 1000));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    const two = (value) => String(value).padStart(2, '0');
    return hours ? two(hours) + ':' + two(minutes) + ':' + two(remainder) : two(minutes) + ':' + two(remainder);
  }

  function updateActivity() {
    const activity = state.activity;
    if (!activity || activity.sessionId !== state.currentId) return;
    const elapsed = Date.now() - activity.startedAt;
    const phases = ['Thinking...', 'Deducing...', 'Plotting...', 'Reasoning...', 'Decoding...'];
    const revealing = state.reveal && state.reveal.sessionId === activity.sessionId;
    $('activity-label').textContent = revealing ? 'Generating...' : phases[Math.floor(elapsed / 2400) % phases.length];
    $('activity-timer').textContent = '[' + formatElapsed(elapsed) + ']';
  }

  function startActivity(sessionId) {
    if (state.activity) clearInterval(state.activity.interval);
    state.lastElapsed = 0;
    state.activity = { sessionId, startedAt: Date.now(), interval: null };
    state.activity.interval = setInterval(updateActivity, 250);
    updateActivity();
  }

  function stopActivity() {
    if (!state.activity) return;
    state.lastElapsed = Date.now() - state.activity.startedAt;
    const session = state.sessions.find((item) => item.id === state.activity.sessionId);
    if (session) session.lastElapsed = state.lastElapsed;
    clearInterval(state.activity.interval);
    state.activity = null;
  }

  function finishTokenReveal(reveal) {
    if (!reveal || reveal.finished) return;
    reveal.finished = true;
    clearInterval(reveal.interval);
    reveal.visibleText = reveal.fullText;
    if (reveal.node && reveal.node.isConnected) reveal.node.textContent = reveal.fullText;
    $('timeline').setAttribute('aria-busy', 'false');
    if (state.reveal === reveal) state.reveal = null;
    reveal.resolve();
  }

  function startTokenReveal(session) {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return null;
    const message = session.messages.slice().reverse().find((item) => item.role === 'assistant' && item.content && item.content.trim());
    if (!message) return null;
    const tokens = message.content.match(/\S+\s*|\s+/g) || [];
    if (tokens.length < 2) return null;
    const reveal = {
      sessionId: session.id,
      message,
      fullText: message.content,
      visibleText: '',
      tokens,
      node: null,
      interval: null,
      resolve: null,
      finished: false
    };
    state.reveal = reveal;
    const promise = new Promise((resolve) => { reveal.resolve = resolve; });
    const rate = Math.min(180, Math.max(45, tokens.length / 8));
    const startedAt = performance.now();
    const tick = () => {
      if (reveal.finished) return;
      const count = Math.min(tokens.length, Math.floor((performance.now() - startedAt) * rate / 1000));
      reveal.visibleText = tokens.slice(0, count).join('');
      if (reveal.node && reveal.node.isConnected) reveal.node.textContent = reveal.visibleText;
      if (count >= tokens.length) finishTokenReveal(reveal);
    };
    reveal.interval = setInterval(tick, 32);
    $('timeline').setAttribute('aria-busy', 'true');
    renderAll();
    updateActivity();
    tick();
    return promise;
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
    startActivity(sessionId);
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
        const terminalError = report.status === 'completed' ? null : (report.error || ('Run ended: ' + report.status));
        target.runs.push({ id: uid(), at: Date.now(), steps: report.steps, usage: report.usage, diagnostics: report.diagnostics, status: report.status, error: terminalError });
        target.status = terminalError ? 'error' : 'complete';
        target.lastError = terminalError;
        target.updatedAt = Date.now();
        if (target.title === 'Untitled task') target.title = prompt.length > 72 ? prompt.slice(0, 69) + '…' : prompt;
        if (!terminalError) {
          const reveal = startTokenReveal(target);
          if (reveal) await reveal;
        }
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
      if (state.reveal && state.reveal.sessionId === sessionId) finishTokenReveal(state.reveal);
      stopActivity();
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
      setContextPanel('run');
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
    $('session-search').addEventListener('input', renderSessions);
    document.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => closeLayer(button.dataset.close)));
    document.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        if (window.matchMedia('(max-width: 980px)').matches) document.body.classList.add('sessions-open');
        $('session-search').focus();
      }
      if (event.key === 'Escape') {
        closeLayer('settings');
        closeLayer('tool');
        document.body.classList.remove('sessions-open');
      }
    });
    $('toggle-context').addEventListener('click', () => document.body.classList.toggle('context-closed'));
    $('close-context').addEventListener('click', () => document.body.classList.add('context-closed'));
    $('toggle-sidebar').addEventListener('click', () => document.body.classList.toggle('sidebar-closed'));
    $('toggle-sessions').addEventListener('click', () => document.body.classList.toggle('sessions-open'));
    $('session-backdrop').addEventListener('click', () => document.body.classList.remove('sessions-open'));

    const contextTabs = Array.from(document.querySelectorAll('[data-context-tab]'));
    contextTabs.forEach((button, index) => {
      button.addEventListener('click', () => setContextPanel(button.dataset.contextTab));
      button.addEventListener('keydown', (event) => {
        let next = null;
        if (event.key === 'ArrowRight') next = (index + 1) % contextTabs.length;
        if (event.key === 'ArrowLeft') next = (index + contextTabs.length - 1) % contextTabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = contextTabs.length - 1;
        if (next !== null) {
          event.preventDefault();
          setContextPanel(contextTabs[next].dataset.contextTab);
          contextTabs[next].focus();
        }
      });
    });
    $('toggle-context-width').addEventListener('click', () => {
      state.contextWide = !state.contextWide;
      document.body.classList.toggle('context-wide', state.contextWide);
      $('toggle-context-width').title = state.contextWide ? 'Collapse dock width' : 'Expand dock';
      saveDockLayout();
    });
    $('files-up').addEventListener('click', async () => {
      if (state.files.parent !== null) {
        await closeFilePreview();
        if (!state.files.selected) loadFiles(state.files.parent);
      }
    });
    $('files-refresh').addEventListener('click', () => loadFiles(state.files.path));
    $('new-file').addEventListener('click', showNewFileForm);
    $('new-file-form').addEventListener('submit', (event) => { event.preventDefault(); createFile(); });
    $('cancel-new-file').addEventListener('click', () => $('new-file-form').classList.add('hidden'));
    $('rename-file').addEventListener('click', showRenameFileForm);
    $('rename-file-form').addEventListener('submit', (event) => { event.preventDefault(); renameSelectedFile(); });
    $('cancel-rename-file').addEventListener('click', () => $('rename-file-form').classList.add('hidden'));
    $('delete-file').addEventListener('click', deleteSelectedFile);
    $('close-file-preview').addEventListener('click', closeFilePreview);
    $('toggle-file-wrap').addEventListener('click', () => {
      $('file-preview-content').classList.toggle('wrap');
      $('file-editor').classList.toggle('wrap');
    });
    $('edit-file').addEventListener('click', () => setFileEditMode(!state.files.editMode));
    $('diff-file').addEventListener('click', () => {
      state.files.showDiff = !state.files.showDiff;
      if (state.files.showDiff) renderFileDiff();
      updateFileViews();
    });
    $('save-file').addEventListener('click', saveFile);
    $('file-editor').addEventListener('input', () => {
      state.files.dirty = true;
      state.files.version += 1;
      setFileStatus('Pending autosave');
      updateFileViews();
      queueFileSave();
    });
    $('file-editor').addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        saveFile();
      }
    });
    window.addEventListener('beforeunload', () => {
      const files = state.files;
      if (!files.selected || !files.dirty || files.saving) return;
      const payload = JSON.stringify({ path: files.selected, content: $('file-editor').value });
      navigator.sendBeacon('/api/v1/files/write', new Blob([payload], { type: 'application/json' }));
    });

    $('terminal-form').addEventListener('submit', (event) => { event.preventDefault(); executeTerminal(); });
    $('terminal-clear').addEventListener('click', () => {
      $('terminal-output').replaceChildren();
      $('terminal-status').className = 'terminal-status';
      $('terminal-status').textContent = 'Ready';
      $('terminal-command').focus();
    });
    $('terminal-command').addEventListener('keydown', (event) => {
      if (!['ArrowUp', 'ArrowDown'].includes(event.key) || !state.terminal.history.length) return;
      event.preventDefault();
      if (event.key === 'ArrowUp') state.terminal.historyIndex = Math.max(0, state.terminal.historyIndex - 1);
      else state.terminal.historyIndex = Math.min(state.terminal.history.length, state.terminal.historyIndex + 1);
      $('terminal-command').value = state.terminal.history[state.terminal.historyIndex] || '';
    });

    $('browser-form').addEventListener('submit', (event) => { event.preventDefault(); navigateBrowser($('browser-url').value, true); });
    $('browser-back').addEventListener('click', () => {
      if (state.browser.current) $('browser-frame').contentWindow.history.back();
    });
    $('browser-forward').addEventListener('click', () => {
      if (state.browser.current) $('browser-frame').contentWindow.history.forward();
    });
    $('browser-reload').addEventListener('click', () => {
      if (!state.browser.current) return;
      try { $('browser-frame').contentWindow.location.reload(); }
      catch (_) { navigateBrowser(state.browser.current, false); }
    });
    $('browser-external').addEventListener('click', () => {
      if (state.browser.current) window.open(state.browser.current, '_blank', 'noopener,noreferrer');
    });
    $('browser-frame').addEventListener('load', () => {
      $('browser-status').classList.add('hidden');
    });
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
    loadDockLayout();
    bindEvents();
    setContextPanel(state.contextPanel, false);
    updateBrowserHistory();
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
