# Icons

Twelve Kadmium-specific glyphs: 24px, 1.75px stroke, round caps and joins, 2–3 units of margin on the 24-unit grid. Each is a single-ink SVG drawn in dark-ground ink `#eceff3` — through `<img>` they suit dark grounds; on a light ground inline the SVG and set `stroke: var(--ink)`, or use `Kadmium.Icon` in React, which draws them in `currentColor`.

- `rod.svg` — one rod through a rail: a shard, a control rod.
- `rods.svg` — four rods: a shard plan, the mark's geometry at icon scale.
- `node-desktop.svg`, `node-laptop.svg`, `node-compact.svg` (Mac Mini and similar), `node-sbc.svg`, `node-nas.svg` — node kinds, as used by `NodeCard`.
- `gpu.svg` — a discrete GPU or a rented GPU instance.
- `cloud-burst.svg` — a cloud with a rod dropping from it: a burst or failover node.
- `memory.svg` — a memory tier.
- `check.svg`, `alert.svg` — the two status glyphs that must match the brand stroke; `alert` always accompanies `warn` colour.

Everything else comes from Lucide at `stroke-width="1.75"`, outline only.
