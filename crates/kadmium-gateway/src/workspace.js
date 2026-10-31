/* Shared feature discovery for the Control Room and the native Companion. */
(function () {
  'use strict';
  const h = React.createElement;
  const features = [
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
    if (feature.href) return { href: feature.href, label: 'Documentation', external: true };
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
  window.KadmiumWorkspace = { Launcher };
})();
