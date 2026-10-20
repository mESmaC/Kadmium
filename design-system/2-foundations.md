# Visual foundations

## Colour

**Graphite** is the ground: four steps, cool-grey rather than black, so cadmium reads as pigment rather than neon. `graphite-0` is the page; `graphite-1` is every raised thing (cards, the sidebar, node tiles); `graphite-2` is a well inside a card (code, table heads, meter tracks, hover); `graphite-3` is pressed. Inputs sit on `graphite-0` even inside a `graphite-1` card, so a form reads as cut into the panel. Separate surfaces with a 1px `line` hairline, never with a shadow alone.

**Cadmium** is one colour with four jobs. Fill (`cadmium`, hover `cadmium-hover`, border `cadmium-edge`, text on it `on-cadmium`): the primary button, the active tab bar, a selected row's marker, local meters and rods. Text, icon and border (`cadmium-text`): links, the shell prompt, the local badge, the focus ring — in the light theme this is a dark gold, because fill-yellow text on white fails. Tint (`cadmium-soft`): the ground behind cadmium text. And the tier ramp (`tier-hot`, `tier-warm`, `tier-cold`): the same hue at three lightnesses for VRAM / RAM / NVMe residency. Budget: one cadmium fill per view, plus any number of cadmium *text* and *rods*.

**Cherenkov** has the identical set (`cherenkov`, `cherenkov-hover`, `cherenkov-text`, `on-cherenkov`, `cherenkov-soft`) and one rule: it is painted only on things that run in the cloud — Kadmium Cloud itself, burst GPUs, spot instances, the managed plane, a failover vault. A screen with no cloud involvement has no cherenkov on it. See `3-open-source-and-cloud.md`.

**Status** is `ok`, `warn`, `danger`, each with a `-soft` ground. `ok` is a teal, deliberately off the red–green axis, so success and failure are never told apart by hue alone; `warn` is orange and lives near cadmium, so a warning always carries its word or the alert icon. Status colours are text, dots and thin fills — never a whole card, never a button fill except `danger` on a confirm.

**Ink** is `ink` for anything a person reads, `ink-muted` for metadata and secondary labels, `ink-faint` for placeholders and disabled labels only. Every pair named in the token notes holds 4.5:1 in both themes; `line-strong`, `cadmium-edge`, `cherenkov` and the focus ring hold 3:1 on `graphite-0` and `graphite-1`.

Don't: gradients of any kind; glows and neon outer shadows; a coloured left border on a card; cadmium and cherenkov in the same control; pure black or pure white surfaces; a third accent for "info".

## Typography

**Archivo** (`sans`) is a grotesk with a slightly rounded, industrial cut — the same temperament as the rod ends. It is set at 650 for display (tight tracking, −0.03 to −0.035 em), 600 for headings and buttons, 400 for text. **JetBrains Mono** (`mono`) carries every number, identifier, command, path, hostname, shard range and hardware line, and all uppercase labels. Both ship as variable woff2 files in `fonts/`; there is no italic.

Hierarchy on a dashboard page: `headline` for the page title, `title` for cards, `body` for cells and prose, `label` (uppercase, mono, 0.08 em tracking) for eyebrows, table heads, meter labels and badges, `metric` for the one big number on a stat tile, `code-sm` where a table is too dense for `code`. On marketing and docs pages: `display-xl` once, `display` for page titles, `body-lg` for the lede, `body` after.

Rules: numbers that will be compared are mono with `font-variant-numeric: tabular-nums`; a unit follows its number with a space ("64 GB", "7 GB/s", "2.3 Gbps"); ranges use an en dash ("layers 12–31"); hardware lines are lowercase, dot-separated ("rtx 5070 · 12 gb · nvme 7 gb/s"); never letter-space sans text, and never set mono text below 12px.

## Space and layout

A 4px grid. Controls are `control-sm` 28 / `control-md` 36 / `control-lg` 44 tall. Cards pad `space-4`, large cards and dialogs `space-5`; cards sit `space-4` apart in a grid, sections `space-6` apart inside a panel, panels `space-8` apart; page gutters are `space-8` on tablet and `space-12` on desktop. The dashboard is dense by design: prefer more cards at `body` size over fewer cards with larger type. The rod pitch — `space-6`, 24px — is the pitch of the pattern and the default rhythm between rods in a `ShardRail` is a multiple of 8.

Topology and plan views are laid out left-to-right in the order shards flow: coordinator, then nodes by pipeline stage, cloud burst nodes last and marked cherenkov.

## Shape and depth

Corners are square-shouldered and soft: `radius-xs` 3 on meter tracks, checkboxes and the active-tab bar; `radius-sm` 6 on buttons, inputs and menu items; `radius-md` 10 on cards, code blocks and popovers; `radius-lg` 16 on dialogs and the topology canvas; `radius-tile` 22 % on app tiles — the YY family corner; `radius-full` only on badges, status dots and rods. Nothing else is a circle.

Elevation is borders first. A resting card is `graphite-1` + 1px `line` + `shadow-sm`. A floating surface (menu, popover, dialog, toast) is `graphite-1` + `shadow-lg`, which already includes the hairline. Nothing glows.

## States

- **Hover**: one step up the graphite scale, or `cadmium-hover` / `cherenkov-hover` on a fill. 120 ms ease.
- **Pressed**: `graphite-3`, no movement.
- **Focus**: `focus-ring` on `:focus-visible` only — never on mouse focus, never removed.
- **Disabled**: opacity 0.5 and `cursor: not-allowed`; never a lighter colour, which would break contrast expectations.
- **Selected**: a `cadmium` bar 3px × `radius-full` (a tab's underline, a list row's marker) or `cadmium-soft` ground with `cadmium-text` — not a cadmium fill.
- **Live states** are the status vocabulary, always with the word: a `Badge` with `dot`.

## Motion

Short and physical. Fills and colours transition in 120 ms; a meter's width in 300 ms ease; a shard handoff on the rail moves the rod, not the colour. Nothing pulses, nothing spins except a determinate transfer, nothing animates on page load. Respect `prefers-reduced-motion` by dropping durations to 0.

## Data visualization

Three forms cover nearly everything: a `Meter` for a level against a capacity (VRAM, UMA, link), a `ShardRail` for how something is split across nodes, and a plain mono table for everything else. Series colours are `cadmium` (local), `cherenkov` (cloud), then the tier ramp when the split is by memory tier; status colours only for status. Axis ticks and legends are `label`; values are `code-sm` tabular. Never a pie, never a 3D anything, never a gradient fill.
