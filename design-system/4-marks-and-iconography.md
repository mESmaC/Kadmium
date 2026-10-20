# Marks, pattern and iconography

## The mark

A head plate and four control rods inserted to different depths — 17, 27, 37 and 47 across a 64-unit box at a 10-unit pitch, the rail at 19, every stroke 7 units with round caps. The rods are `cadmium`; the rail is ink (`ink` on dark grounds, `#15181c` on light). It is also an asymmetric shard plan: four nodes, four shares. Files in `assets/Logos`:

- `kadmium-mark-dark.svg` / `kadmium-mark-light.svg` — the two-colour mark for dark and light grounds.
- `kadmium-mark-mono-dark.svg` / `kadmium-mark-mono-light.svg` — single ink, for embossing, favicons on a coloured ground, partner lockups and anywhere cadmium is unavailable.
- `kadmium-wordmark-dark.svg` / `kadmium-wordmark-light.svg` — the mark scaled to the cap height, a 22-unit gap, then *Kadmium* set in Archivo 650 as outlines (no font dependency). Use it for the site header, the README, documents.
- `kadmium-tile-graphite.svg` / `kadmium-tile-cadmium.svg` — app tiles, the YY family format: a square with a 22 % corner radius and the mark at 60 % of the side. Graphite is the default; cadmium is for app stores and avatars where the tile must carry the colour.
- `kadmium-favicon.svg` — a three-rod cut on the graphite tile, drawn for 16–32 px.

In a React page, `Kadmium.Mark` renders the same geometry inline, so it picks up `currentColor` for the rail and the theme's cadmium for the rods.

**Clear space** is one rod pitch — 10/64 of the mark's height — on every side; nothing sits closer. **Minimum size** is 20px for the mark, 16px for the favicon cut, 96px wide for the wordmark. **Don't** recolour the rods anything but cadmium or the ink, rotate the mark (rods hang; they never point up or sideways), add a fifth rod, round the rail into a ring, outline the mark, put it on a photograph, or set the wordmark in any face but Archivo 650 at the cap-height relation the file fixes.

**Family.** Kadmium is a YY (YinYangAI) brand; the parent's own mark is in `assets/Parent Company Logo` for reference. The family resemblance is deliberate and limited to three things: the app-tile format (a rounded square, 22 % radius, charcoal or the accent), the round-capped stroke language, and the one-accent-on-charcoal palette structure. YY's mark, its green and its interlaced strokes are never used on Kadmium surfaces, and Kadmium's rods never appear inside YY's tile.

## The pattern

The rod lattice is the only pattern: a horizontal rail with rods hanging from it at a `space-6` (24px) pitch, at unequal depths, in `cadmium` — or cut out of a cadmium block in `on-cadmium` — with at most one or two rods in `cherenkov` where a composition is about a mixed cluster. Use it as a band across a hero or a section break, bleeding off one edge; a dozen to sixty rods, never a field. Depths should look like a real plan (a few long, most medium, one or two short), not a sine wave and not random noise. The cover is the reference composition. Nothing else — no grids of dots, no circuit traces, no atoms, no glows.

## Icons

Icons are 24 px, 1.75 px stroke, round caps and joins, drawn on a 24-unit grid with 2–3 units of margin — the stroke weight and cap of the mark at icon scale. Twelve Kadmium-specific glyphs are in `assets/Icons`, all single-ink SVGs drawn in dark-ground ink (`#eceff3`): `rod`, `rods` (a shard plan), `node-desktop`, `node-laptop`, `node-compact` (Mac Mini and similar), `node-sbc`, `node-nas`, `gpu`, `cloud-burst`, `memory`, `check`, `alert`. On a light ground, inline the SVG and set `stroke: var(--ink)`, or use `Kadmium.Icon` in React, which draws them in `currentColor`.

For every other glyph the system uses **Lucide** (ISC licence, 24 px, 2 px stroke by default): set `stroke-width="1.75"` to match. This is an intentional substitution for a from-scratch brand, not a copy of anyone's set; keep to outline icons, never filled, never two-tone, never emoji.

Icon colour follows the text it sits with: `ink` beside `ink`, `ink-muted` beside metadata, `cadmium-text` / `cherenkov-text` only when the icon itself means local or cloud, status colours only on status. Icons inside buttons are `icon-sm` (16 px); in inputs and node cards `icon-md` (20 px); standalone `icon-lg` (24 px).
