/* Shared feature discovery for the Control Room and the native Companion. */
(function () {
  'use strict';
  const h = React.createElement;
  const features = [
    { title: 'Prepare a local model', description: 'Choose a GGUF or SafeTensors model, prepare it without quality loss, and run it on this computer.', href: '/condenser/#models', group: 'Build', keywords: 'import weights checkpoint library bring your own' },
    { title: 'Send a model to a coordinator', description: 'Send a prepared model to a Kadmium coordinator in verified, resumable chunks.', href: '/condenser/#models-cloud', group: 'Build', keywords: 'upload transport remote premium hosted gpu' },
    { title: 'Launch Condenser', description: 'Chat, code, and run tools with your chosen model.', href: '/condenser/', group: 'Build', keywords: 'agent inference model context tokens' },
    { title: 'Cluster overview', description: 'Machines, workloads, and recent activity.', path: 'overview', local: 'home', group: 'Monitor', keywords: 'home dashboard status' },
    { title: 'Machines', description: 'Join a computer, inspect hardware, or pause sharing.', path: 'nodes', local: 'cluster', group: 'Monitor', keywords: 'add pair node worker lan' },
    { title: 'Boards & devices', description: 'Attach Macs, Raspberry Pi, USB boards, and phone farms.', path: 'platforms', group: 'Monitor', keywords: 'm4 cm4 k230 tegra hardware' },
    { title: 'Models & shards', description: 'Load a model and distribute it across your machines.', path: 'models', local: 'models', group: 'Build', keywords: 'inference import gguf safetensors weights' },
    { title: 'API & playground', description: 'Try a prompt or connect OpenAI, Anthropic, and MCP clients.', path: 'gateway', group: 'Build', keywords: 'chat inference tokens endpoint integrations' },
    { title: 'Training & checkpoints', description: 'Fine-tune, save progress, and resume a training run.', path: 'training', group: 'Build', keywords: 'lora loss learning restore' },
    { title: 'Reinforcement learning', description: 'Run verified attempts and explore Dream-RSI replay.', path: 'rl', group: 'Build', keywords: 'grpo reward dream rsi' },
    { title: 'Shared storage', description: 'Manage the pooled bucket and transfers between machines.', path: 'data/bucket', group: 'Manage', keywords: 'files disk data nas fts3' },
    { title: 'Cluster slices', description: 'Create smaller clusters with their own models and limits.', path: 'slices', group: 'Manage', keywords: 'subcluster quota partition' },
    { title: 'Kadmium Cloud', description: 'Manage burst GPUs, failover protection, and spending.', path: 'cloud', group: 'Manage', keywords: 'shield standby h100 spot budget' },
    { title: 'Earn from idle time', description: 'Choose how your hardware participates in the Supply Grid.', path: 'cloud/supply', local: 'earn', group: 'Manage', keywords: 'rent lease credits supply marketplace' },
    { title: 'Tools & tests', description: 'Test prompts, transfers, model sizing, and hardware bridges.', companion: 'tools', group: 'Manage', keywords: 'diagnostics benchmark exl3' },
    { title: 'Settings', description: 'Appearance, connection, versions, and updates.', path: 'settings', local: 'settings', group: 'Manage', keywords: 'theme light dark self hosted' },
    { title: 'Documentation', description: 'Setup guides, architecture, and the API reference.', href: '/docs', group: 'Learn', keywords: 'help getting started cli condenser agent workspace' }
  ];
  function destination(feature, companion) {
    if (feature.href) return { href: feature.href, label: feature.href.startsWith('/condenser') ? 'Condenser' : 'Documentation', external: !feature.href.startsWith('/condenser') };
    if (companion && (feature.local || feature.companion)) return { href: '#' + (feature.local || feature.companion), label: 'Companion' };
    if (feature.companion) return { href: '/companion#' + feature.companion, label: 'Companion', external: true };
    return { href: '/#/' + feature.path, label: 'Control Room', external: companion };
  }
  function Launcher(props) {
    const [open, setOpen] = React.useState(false);
    const [query, setQuery] = React.useState('');
    const dialog = React.useRef(null);
    const input = React.useRef(null);
    const trigger = React.useRef(null);
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const results = features.filter(f => terms.every(t => (f.title + ' ' + f.description + ' ' + f.group + ' ' + f.keywords).toLowerCase().includes(t)));
    const close = () => setOpen(false);
    React.useEffect(() => {
      const shortcut = e => {
        if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
          // Do not steal shortcuts from another active modal.
          if (!open && document.querySelector('dialog[open], [aria-modal="true"]')) return;
          e.preventDefault(); setOpen(v => !v);
        }
      };
      document.addEventListener('keydown', shortcut);
      return () => document.removeEventListener('keydown', shortcut);
    }, [open]);
    React.useEffect(() => {
      if (!open) return undefined;
      const previous = document.activeElement;
      const el = dialog.current;
      el.showModal();
      input.current.focus();
      return () => {
        el.close();
        if (previous && previous.isConnected) previous.focus();
        else if (trigger.current) trigger.current.focus();
      };
    }, [open]);
    function choose(feature) {
      const target = destination(feature, props.companion);
      close(); setQuery('');
      if (props.onNavigate) props.onNavigate(target);
      else if (target.external) window.open(target.href, '_blank', 'noopener');
      else window.location.href = target.href;
    }
    function onKey(e) {
      const buttons = Array.from(dialog.current.querySelectorAll('.kw-result'));
      if (e.target === input.current && e.key === 'Enter' && results.length) { e.preventDefault(); choose(results[0]); return; }
      if (!['ArrowDown', 'ArrowUp'].includes(e.key) || !buttons.length) return;
      if (e.target !== input.current && !buttons.includes(e.target)) return;
      e.preventDefault();
      const index = buttons.indexOf(document.activeElement);
      if (e.key === 'ArrowUp' && index <= 0) input.current.focus();
      else buttons[(index + (e.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length].focus();
    }
    return h(React.Fragment, null,
      h('button', { ref: trigger, className: 'kw-launch', type: 'button', onClick: () => { setQuery(''); setOpen(true); }, 'aria-label': 'Find a feature', 'aria-haspopup': 'dialog', 'aria-keyshortcuts': 'Control+k Meta+k', title: 'Find a feature (Ctrl+K / ⌘K)' },
        h('svg', { viewBox: '0 0 24 24', width: 17, height: 17, fill: 'none', stroke: 'currentColor', strokeWidth: 1.75, 'aria-hidden': true }, h('circle', { cx: 10.5, cy: 10.5, r: 6.5 }), h('path', { d: 'm16 16 5 5' })),
        h('span', null, 'Find a feature'), h('kbd', { 'aria-hidden': true }, /Mac/.test(navigator.platform) ? '⌘ K' : 'Ctrl K')),
      open ? ReactDOM.createPortal(h('dialog', { ref: dialog, className: 'kw-dialog', 'aria-labelledby': 'kw-title', onCancel: close, onKeyDown: onKey, onClick: e => { if (e.target === dialog.current) { const r = dialog.current.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) close(); } } },
        h('header', { className: 'kw-dialog-head' }, h('div', null, h('span', { className: 'kw-eyebrow' }, 'Your compute workspace'), h('h2', { id: 'kw-title' }, 'What would you like to do?')), h('button', { type: 'button', className: 'kw-close', onClick: close, 'aria-label': 'Close feature finder' }, '×')),
        h('label', { className: 'kw-search' }, h('span', null, 'Search features'), h('input', { ref: input, type: 'search', value: query, onChange: e => setQuery(e.target.value), placeholder: 'Try “train”, “MCP”, or “cloud”…', autoComplete: 'off', 'aria-controls': 'kw-results' })),
        h('p', { className: 'kw-result-count', role: 'status' }, results.length + ' feature' + (results.length === 1 ? '' : 's') + (props.companion ? ' · Control Room features open in your browser' : ' · Choose a destination')),
        h('div', { className: 'kw-results', id: 'kw-results' }, results.length ? results.map(f => {
          const target = destination(f, props.companion);
          return h('button', { key: f.title, className: 'kw-result', type: 'button', onClick: () => choose(f) }, h('span', { className: 'kw-result-copy' }, h('span', { className: 'kw-result-title' }, f.title), h('span', { className: 'kw-result-description' }, f.description)), h('span', { className: 'kw-destination' }, target.external ? target.label + ' ↗' : f.group));
        }) : h('div', { className: 'kw-no-results' }, h('strong', null, 'No matching features'), h('p', null, 'Try a task like “model”, “storage”, or “training”.'), h('button', { type: 'button', className: 'kw-close', onClick: () => { setQuery(''); input.current.focus(); } }, 'Clear search'))),
        h('footer', { className: 'kw-dialog-foot' }, '↑ ↓ to navigate · Enter to open · Esc to close')), document.body) : null);
  }
  function ModelSetup() {
    const local = ['127.0.0.1', 'localhost'].includes(location.hostname);
    const [received, setReceived] = React.useState(null);
    React.useEffect(() => {
      if (!local) return undefined;
      let live = true;
      fetch('/api/v1/model-transport').then(r => r.ok ? r.json() : null).then(d => { if (live && d) setReceived(d); }).catch(() => {});
      return () => { live = false; };
    }, [local]);
    const served = new Set(received ? received.served : []);
    const destinations = [
      { id: 'local', title: 'On this computer', badge: h('span', { className: 'k-pill' }, 'Private'), body: 'Runs through Ollama on this machine. Nothing leaves your network. Best when this computer has enough memory for the model.', href: '/condenser/#models', action: 'Prepare a local model' },
      { id: 'cloud', title: 'Remote coordinator', badge: null, body: 'Send the prepared model to a Kadmium coordinator you run, in verified, resumable chunks. It loads in that coordinator\u2019s Ollama at full context. Kadmium Cloud, the planned premium service, will use the same flow.', href: '/condenser/#models-cloud', action: 'Send to a coordinator' }
    ];
    return h('section', { className: 'kw-model-setup', 'aria-labelledby': 'kw-model-title' },
      h('div', { className: 'kw-model-head' },
        h('span', { className: 'k-eyebrow' }, 'Your models'),
        h('h2', { id: 'kw-model-title' }, 'Run your own model'),
        h('p', null, 'Choose a model file, let Condenser prepare it, then run it here or on a Kadmium coordinator. Weights are never quantized, pruned, or shortened, and every model starts with its full context and output.'),
        h('ol', { className: 'k-steps', 'aria-label': 'How it works' },
          h('li', null, 'Choose a GGUF or SafeTensors model'),
          h('li', null, 'Prepare: verified, 4 KiB-aligned shards'),
          h('li', null, 'Run on this computer or a remote coordinator'))),
      local ? h('div', { className: 'kw-destinations' }, destinations.map(d => h('article', { key: d.id, className: 'kw-destination-card ' + d.id },
        h('header', null, h('h3', null, d.title), d.badge),
        h('p', null, d.body),
        h('a', { className: 'kw-cta', href: d.href }, d.action, h('span', { 'aria-hidden': 'true' }, ' →'))))) : null,
      local && received && received.checkpoints.length ? h('div', { className: 'kw-received' },
        h('h3', null, 'Models on this coordinator'),
        h('ul', null, received.checkpoints.map(m => h('li', { key: m.id },
          h('span', null, m.name), h('span', { className: 'k-pill ' + (served.has('kadmium-' + m.id + ':latest') ? 'ok' : '') }, served.has('kadmium-' + m.id + ':latest') ? 'Serving' : 'Ready to load'))))) : null,
      h('p', { className: 'kw-model-foot' }, local
        ? h(React.Fragment, null, 'Lower context or output per task in Condenser → Model & quality; each limit explains what it costs. ', h('a', { href: '/condenser/' }, 'Launch Condenser'))
        : h('span', { role: 'status' }, 'Model files live on the coordinator computer. Open Kadmium there at http://127.0.0.1:' + location.port + '/ to prepare models and launch Condenser.')));
  }
  window.KadmiumWorkspace = { Launcher, ModelSetup };
})();
