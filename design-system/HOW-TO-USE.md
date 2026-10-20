# Kadmium design system — how to use this folder

Open `demo.html` in a browser to see everything running (no build step, no network). Then:

**In any web page or the `kadmium-gateway` dashboard**

```html
<html data-theme="dark">                     <!-- or "light"; dark is the fallback -->
<link rel="stylesheet" href="tokens.css">     <!-- compiled from tokens.json: every colour per theme, spacing, radius, sizes, fonts, type styles -->
<link rel="stylesheet" href="components/bundle.css">
<script src="components/lib/react.production.min.js"></script>
<script src="components/lib/react-dom.production.min.js"></script>
<script src="components/bundle.js"></script>  <!-- attaches window.Kadmium -->
<script>
  const { Button, NodeCard, ShardRail } = window.Kadmium;
  ReactDOM.createRoot(root).render(React.createElement(Button, { variant: 'primary' }, 'Start node'));
</script>
```

Keep `fonts/` next to `tokens.css` (its `@font-face` rules point there). Props for every component are documented in `components/index.d.ts`; usage rules in each `components/<Name>/README.md`.

**Tokens only (no React)** — link `tokens.css` and use the variables (`var(--cadmium)`, `var(--space-4)`, `var(--radius-md)`) and the type classes (`.display`, `.body`, `.label`, `.code`). `tokens.json` is the source of truth; regenerate `tokens.css` from it when you change a value (the live design system does this for you).

**Marks and icons** — `assets/Logos` (SVG, dark/light-ground versions, wordmark, app tiles, favicon) and `assets/Icons` (24 px, 1.75 stroke). Rules in `4-marks-and-iconography.md`.

**The brand book** is `README.md` plus the four numbered sections. The live, editable copy of all of this is the Kadmium design system in Claude; this folder is a snapshot of it.
