# Meter

A level against a capacity — VRAM, UMA, RAM, NVMe, a link — as a 6px track with a cadmium (local) or cherenkov (cloud) fill and a tabular mono value.

You provide `label`, `value`, `max` and a `unit`; `tone="cloud"` for rented hardware; `criticalAt` (a fraction, e.g. 0.95 for VRAM) or `critical` to turn the fill `danger`. Stack meters inside a `NodeCard` or a stat tile, `space-2` apart, longest-lived tier first (VRAM, then RAM, then NVMe). A meter is for one quantity; for how something splits across nodes, use a `ShardRail`.
