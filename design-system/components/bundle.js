/* @ds-bundle: {"format":4,"namespace":"Kadmium","components":[{"name":"Button"},{"name":"Badge"},{"name":"Meter"},{"name":"NodeCard"},{"name":"ShardRail"},{"name":"PlanCard"},{"name":"CodeBlock"},{"name":"Input"},{"name":"Tabs"},{"name":"Mark"}]} */
(function () {
  'use strict';
  var React = window.React;
  if (!React) { return; }
  var h = React.createElement;

  function cx() { var out = []; for (var i = 0; i < arguments.length; i++) { if (arguments[i]) { out.push(arguments[i]); } } return out.join(' '); }
  function omit(obj, keys) { var out = {}; for (var k in obj) { if (Object.prototype.hasOwnProperty.call(obj, k) && keys.indexOf(k) < 0) { out[k] = obj[k]; } } return out; }
  var uid = 0;

  /* ---------- icons (24px, 1.75 stroke, round caps — the same drawings as assets/Icons) ---------- */
  var ICONS = {
    desktop: [['rect', { x: 3, y: 4, width: 18, height: 12, rx: 2 }], ['path', { d: 'M8 20h8M12 16v4' }]],
    laptop: [['rect', { x: 4, y: 5, width: 16, height: 10, rx: 2 }], ['path', { d: 'M2 19h20' }]],
    compact: [['rect', { x: 3, y: 8, width: 18, height: 8, rx: 2 }], ['path', { d: 'M7 12h.01M15 12h3' }]],
    sbc: [['rect', { x: 3, y: 5, width: 14, height: 14, rx: 2 }], ['path', { d: 'M17 8h4M17 12h4M17 16h4' }], ['rect', { x: 7, y: 9, width: 6, height: 6, rx: 1 }]],
    nas: [['rect', { x: 4, y: 3, width: 16, height: 8, rx: 2 }], ['rect', { x: 4, y: 13, width: 16, height: 8, rx: 2 }], ['path', { d: 'M8 7h.01M8 17h.01' }]],
    gpu: [['rect', { x: 2, y: 6, width: 20, height: 11, rx: 2 }], ['circle', { cx: 14, cy: 11.5, r: 3 }], ['path', { d: 'M5 17v3M9 17v3' }]],
    cloud: [['path', { d: 'M7 17a4 4 0 0 1-.6-7.95A6 6 0 0 1 18 8a4.5 4.5 0 0 1 0 9' }], ['path', { d: 'M12 12v9' }]],
    rod: [['path', { d: 'M5 6h14M12 3v15' }]],
    rods: [['path', { d: 'M4 7h16M8 4v10M12 4v16M16 4v8' }]],
    memory: [['rect', { x: 3, y: 7, width: 18, height: 10, rx: 2 }], ['path', { d: 'M7 7v10M11 7v10M15 7v10' }]],
    alert: [['path', { d: 'M12 4l9 16H3z' }], ['path', { d: 'M12 10v4M12 17h.01' }]],
    check: [['path', { d: 'M5 12.5l4.5 4.5L19 7.5' }]],
    copy: [['rect', { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ['path', { d: 'M5 15V6a2 2 0 0 1 2-2h9' }]]
  };
  function Icon(name, extra) {
    var parts = ICONS[name] || ICONS.rods;
    return h('svg', Object.assign({ viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.75, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': 'true' }, extra || {}),
      parts.map(function (p, i) { return h(p[0], Object.assign({ key: i }, p[1])); }));
  }

  /* ---------- Button ---------- */
  function Button(props) {
    var variant = props.variant || 'quiet', size = props.size || 'md';
    var rest = omit(props, ['variant', 'size', 'icon', 'children', 'className']);
    return h('button', Object.assign({ type: 'button' }, rest, { className: cx('kd-btn', 'kd-btn-' + variant, size !== 'md' && 'kd-btn-' + size, props.className) }),
      props.icon ? h('span', { className: 'kd-btn-icon', 'aria-hidden': 'true' }, typeof props.icon === 'string' ? Icon(props.icon) : props.icon) : null,
      props.children);
  }

  /* ---------- Badge ---------- */
  function Badge(props) {
    var tone = props.tone || 'neutral';
    return h('span', { className: cx('kd-badge', tone !== 'neutral' && 'kd-badge-' + tone, props.dot && 'kd-badge-dot', props.className), title: props.title }, props.children);
  }

  /* ---------- Meter ---------- */
  function Meter(props) {
    var max = props.max > 0 ? props.max : 100, value = Math.max(0, Math.min(props.value || 0, max));
    var pct = Math.round((value / max) * 100);
    var critical = props.critical != null ? props.critical : (props.criticalAt != null && value / max >= props.criticalAt);
    var text = props.format ? props.format(value, max) : (fmt(value) + ' / ' + fmt(max) + (props.unit ? ' ' + props.unit : ''));
    return h('div', { className: cx('kd-meter', props.tone === 'cloud' && 'kd-meter-cloud', critical && 'kd-meter-critical', props.className), role: 'meter', 'aria-label': props.label, 'aria-valuemin': 0, 'aria-valuemax': max, 'aria-valuenow': value },
      h('span', { className: 'kd-meter-label' }, props.label),
      h('span', { className: 'kd-meter-track' }, h('span', { className: 'kd-meter-fill', style: { width: pct + '%' } })),
      h('span', { className: 'kd-meter-value' }, text));
  }
  function fmt(n) { return Number.isInteger(n) ? String(n) : (Math.round(n * 10) / 10).toString(); }

  /* ---------- NodeCard ---------- */
  var STATUS = {
    resident: { tone: 'ok', label: 'resident' },
    training: { tone: 'ok', label: 'training' },
    prefetching: { tone: 'cloud', label: 'prefetching' },
    joining: { tone: 'neutral', label: 'joining' },
    yielded: { tone: 'warn', label: 'yielded' },
    throttled: { tone: 'warn', label: 'thermal' },
    offline: { tone: 'danger', label: 'offline' },
    faulted: { tone: 'danger', label: 'faulted' }
  };
  function NodeCard(props) {
    var placement = props.placement || 'local', status = STATUS[props.status || 'resident'] || { tone: 'neutral', label: props.status };
    var kind = props.kind || (placement === 'cloud' ? 'cloud' : 'desktop');
    var meters = props.meters || [];
    return h('article', { className: cx('kd-node', 'kd-node-' + placement, (props.status === 'offline') && 'kd-node-offline', props.className), 'aria-label': props.name },
      h('header', { className: 'kd-node-head' },
        h('span', { className: 'kd-node-icon' }, Icon(kind)),
        h('div', { className: 'kd-node-title' }, h('div', { className: 'kd-node-name' }, props.name), props.hardware ? h('div', { className: 'kd-node-hw' }, props.hardware) : null),
        h(Badge, { tone: placement === 'cloud' ? 'cloud' : 'local' }, placement === 'cloud' ? 'cloud' : 'local')),
      meters.length ? h('div', { className: 'kd-node-meters' }, meters.map(function (m, i) { return h(Meter, Object.assign({ key: i, tone: m.tone || (placement === 'cloud' ? 'cloud' : 'local') }, m)); })) : null,
      props.children,
      h('footer', { className: 'kd-node-foot' },
        h(Badge, { tone: status.tone, dot: true }, status.label),
        props.shards ? h('span', { className: 'kd-node-shards' }, props.shards) : null));
  }

  /* ---------- ShardRail ---------- */
  function ShardRail(props) {
    var nodes = props.nodes || [], pitch = props.pitch || 56, depth = props.depth || 96, pad = 20;
    var width = pad * 2 + Math.max(1, nodes.length - 1) * pitch, railY = 16, labelY = railY + depth + 24;
    var height = labelY + (props.showShare === false ? 6 : 20);
    var maxShare = nodes.reduce(function (m, n) { return Math.max(m, n.share || 0); }, 0) || 1;   // the deepest rod reaches `depth`
    return h('svg', { className: cx('kd-rail', props.className), viewBox: '0 0 ' + width + ' ' + height, width: width, height: height, role: 'img', 'aria-label': props.label || 'Shard plan' },
      props.label ? h('title', null, props.label) : null,
      h('line', { className: 'kd-rail-bar', x1: 4, y1: railY, x2: width - 4, y2: railY }),
      nodes.map(function (n, i) {
        var x = pad + i * pitch, share = Math.max(0, Math.min(1, n.share || 0));
        var y2 = railY + (share > 0 ? Math.max(8, (share / maxShare) * depth) : 6);
        return h('g', { key: n.id || i },
          h('line', { className: cx('kd-rail-rod', n.placement === 'cloud' && 'kd-rail-rod-cloud', share === 0 && 'kd-rail-rod-idle'), x1: x, y1: railY - 8, x2: x, y2: y2 }),
          h('text', { className: 'kd-rail-label', x: x, y: labelY, textAnchor: 'middle' }, n.name),
          props.showShare === false ? null : h('text', { className: 'kd-rail-pct', x: x, y: labelY + 16, textAnchor: 'middle' }, Math.round(share * 100) + '%'));
      }));
  }

  /* ---------- PlanCard ---------- */
  function PlanCard(props) {
    var cloud = props.product === 'cloud';
    return h('section', { className: cx('kd-plan', cloud ? 'kd-plan-cloud' : 'kd-plan-self', props.className), 'aria-label': props.name },
      h('div', { className: 'kd-plan-eyebrow' }, props.eyebrow || (cloud ? 'Kadmium Cloud' : 'Self-hosted')),
      h('h3', { className: 'kd-plan-name', style: { margin: 0 } }, props.name),
      h('div', { className: 'kd-plan-price' }, props.price, props.period ? h('small', null, props.period) : null),
      props.summary ? h('p', { className: 'kd-plan-summary', style: { margin: 0 } }, props.summary) : null,
      props.features && props.features.length ? h('ul', { className: 'kd-plan-features' }, props.features.map(function (f, i) { return h('li', { key: i }, f); })) : null,
      props.cta ? h(Button, { variant: cloud ? 'cloud' : 'primary', size: 'lg', onClick: props.onSelect }, props.cta) : null);
  }

  /* ---------- CodeBlock ---------- */
  var KW = {
    rust: 'as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while'.split(' '),
    toml: ['true', 'false'], json: ['true', 'false', 'null'], shell: ['cargo', 'kadmium', 'cadmium', 'run', 'test', 'curl', 'export', 'cd', 'sudo', 'docker', 'ssh'], text: []
  };
  function tokenize(line, lang) {
    var out = [], i = 0, re;
    if (lang === 'shell') {
      re = /(\s+)|(^[$>]\s)|(#.*$)|("(?:[^"\\]|\\.)*"|'[^']*')|(--?[A-Za-z][\w-]*)|(\b\d[\d.,:]*\b)|([^\s]+)/g;
    } else if (lang === 'rust') {
      re = /(\s+)|(\/\/.*$)|("(?:[^"\\]|\\.)*")|(\b\d[\d_.]*(?:[iuf]\d+)?\b)|([A-Za-z_]\w*!?)|([^\sA-Za-z_\d]+)/g;
    } else if (lang === 'toml' || lang === 'json') {
      re = /(\s+)|(#.*$)|("(?:[^"\\]|\\.)*"|'[^']*')|(-?\b\d[\d_.eE+-]*\b)|([A-Za-z_][\w.-]*)|([^\s]+)/g;
    } else {
      return [{ t: '', v: line }];
    }
    var m;
    while ((m = re.exec(line)) !== null) {
      if (m.index !== i) { out.push({ t: '', v: line.slice(i, m.index) }); }
      i = re.lastIndex;
      var v = m[0], t = '';
      if (lang === 'shell') {
        if (m[2]) { t = 'p'; } else if (m[3]) { t = 'c'; } else if (m[4]) { t = 's'; } else if (m[5]) { t = 'k'; } else if (m[6]) { t = 'n'; }
        else if (m[7] && KW.shell.indexOf(v) >= 0 && (out.length === 0 || out.every(function (o) { return o.t === 'p' || /^\s*$/.test(o.v); }))) { t = 'k'; }
      } else if (lang === 'rust') {
        if (m[2]) { t = 'c'; } else if (m[3]) { t = 's'; } else if (m[4]) { t = 'n'; } else if (m[5] && (KW.rust.indexOf(v) >= 0 || /!$/.test(v))) { t = 'k'; }
      } else {
        if (m[2]) { t = 'c'; } else if (m[3]) { t = 's'; } else if (m[4]) { t = 'n'; } else if (m[5] && KW[lang].indexOf(v) >= 0) { t = 'k'; }
      }
      out.push({ t: t, v: v });
    }
    if (i < line.length) { out.push({ t: '', v: line.slice(i) }); }
    return out;
  }
  function CodeBlock(props) {
    var lang = props.language || 'shell', code = props.code || '';
    var copied = React.useState(false), isCopied = copied[0], setCopied = copied[1];
    function onCopy() {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(code).then(function () { setCopied(true); setTimeout(function () { setCopied(false); }, 1500); }); }
      } catch (e) { /* clipboard unavailable: the button is still a visible affordance */ }
    }
    var lines = code.replace(/\n$/, '').split('\n');
    return h('figure', { className: cx('kd-code', props.className), style: { margin: 0 } },
      (props.title || props.copy !== false) ? h('figcaption', { className: 'kd-code-head' },
        h('span', null, props.title || lang),
        props.copy !== false ? h(Button, { variant: 'ghost', size: 'sm', icon: isCopied ? 'check' : 'copy', onClick: onCopy, 'aria-label': 'Copy code' }, isCopied ? 'Copied' : 'Copy') : null) : null,
      h('pre', null, h('code', null, lines.map(function (line, li) {
        var toks = tokenize(line, lang);
        return h(React.Fragment, { key: li }, toks.map(function (tk, ti) { return tk.t ? h('span', { key: ti, className: 'kd-tok-' + tk.t }, tk.v) : tk.v; }), li < lines.length - 1 ? '\n' : null);
      }))));
  }

  /* ---------- Input ---------- */
  function Input(props) {
    var idRef = React.useRef(null); if (!idRef.current) { idRef.current = props.id || ('kd-in-' + (++uid)); }
    var id = idRef.current, hintId = id + '-hint';
    var rest = omit(props, ['label', 'hint', 'error', 'mono', 'prefix', 'className', 'id']);
    var message = props.error || props.hint;
    return h('div', { className: cx('kd-field', props.error && 'kd-field-error', props.mono && 'kd-input-mono', props.disabled && 'kd-field-disabled', props.className) },
      props.label ? h('label', { className: 'kd-field-label', htmlFor: id }, props.label) : null,
      h('div', { className: 'kd-input-wrap' },
        props.prefix ? h('span', { className: 'kd-input-prefix', 'aria-hidden': 'true' }, props.prefix) : null,
        h('input', Object.assign({ type: 'text' }, rest, { id: id, className: 'kd-input', 'aria-invalid': props.error ? 'true' : undefined, 'aria-describedby': message ? hintId : undefined }))),
      message ? h('div', { className: 'kd-field-hint', id: hintId }, message) : null);
  }

  /* ---------- Tabs ---------- */
  function Tabs(props) {
    var items = props.items || [];
    return h('div', { className: cx('kd-tabs', props.className), role: 'tablist', 'aria-label': props.label },
      items.map(function (it) {
        var selected = it.id === props.value;
        return h('button', { key: it.id, type: 'button', role: 'tab', className: 'kd-tab', 'aria-selected': selected ? 'true' : 'false', tabIndex: selected ? 0 : -1,
          onClick: function () { if (props.onChange) { props.onChange(it.id); } },
          onKeyDown: function (e) {
            var idx = items.indexOf(it), next = null;
            if (e.key === 'ArrowRight') { next = items[(idx + 1) % items.length]; } else if (e.key === 'ArrowLeft') { next = items[(idx - 1 + items.length) % items.length]; }
            if (next && props.onChange) { e.preventDefault(); props.onChange(next.id); var el = e.currentTarget.parentNode.children[items.indexOf(next)]; if (el) { el.focus(); } }
          } },
          it.label, it.count != null ? h('span', { className: 'kd-tab-count' }, it.count) : null);
      }));
  }

  /* ---------- Mark ---------- */
  function Mark(props) {
    var size = props.size || 24;
    return h('svg', { className: cx('kd-mark', props.tone === 'mono' && 'kd-mark-mono', props.className), viewBox: '0 0 64 64', width: size, height: size, fill: 'none', strokeWidth: 7, strokeLinecap: 'round', role: 'img', 'aria-label': props.title || 'Kadmium' },
      h('line', { className: 'kd-mark-rod', x1: 17, y1: 10, x2: 17, y2: 40 }),
      h('line', { className: 'kd-mark-rod', x1: 27, y1: 10, x2: 27, y2: 54 }),
      h('line', { className: 'kd-mark-rod', x1: 37, y1: 10, x2: 37, y2: 30 }),
      h('line', { className: 'kd-mark-rod', x1: 47, y1: 10, x2: 47, y2: 46 }),
      h('line', { className: 'kd-mark-rail', x1: 11, y1: 19, x2: 53, y2: 19 }));
  }

  /* ---------- Icon (the assets/Icons glyphs, drawn inline in currentColor) ---------- */
  function IconComponent(props) {
    var size = props.size || 24;
    return Icon(props.name, { width: size, height: size, className: cx('kd-icon', props.className), 'aria-label': props.title, 'aria-hidden': props.title ? undefined : 'true', role: props.title ? 'img' : undefined });
  }

  var api = { Button: Button, Badge: Badge, Meter: Meter, NodeCard: NodeCard, ShardRail: ShardRail, PlanCard: PlanCard, CodeBlock: CodeBlock, Input: Input, Tabs: Tabs, Mark: Mark, Icon: IconComponent };
  window.Kadmium = Object.assign(window.Kadmium || {}, api);
})();
