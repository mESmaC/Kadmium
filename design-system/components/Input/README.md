# Input

A labelled text field cut into the panel: `graphite-0` on a `graphite-1` card, `line-strong` border, cadmium focus ring. `mono` for hostnames, ports, paths, model ids and budgets; a `prefix` (`quic://`, `$`) for a fixed scheme or unit.

You provide `label` (always — placeholders are not labels), an optional `hint`, an `error` that replaces the hint and marks the field invalid, and any input attributes. The component wires `id`, `aria-describedby` and `aria-invalid`. Keep hints to one sentence in `caption` size.
