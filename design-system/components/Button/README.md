# Button

Quiet is the default; `primary` (cadmium) at most once per view, for the thing the view is for. `cloud` (cherenkov) only triggers a Kadmium Cloud action — attach a burst GPU, start a cluster, add the failover shield. `ghost` for toolbars and card headers, `danger` for destructive actions behind a confirmation.

You provide the label (verb first, sentence case: "Start node", "Attach burst GPU"), optional `icon` (a built-in glyph name or your own 16px SVG), `size` and any button attributes. Labels never wrap; keep them to three words. A button never carries a status colour — `ok` and `warn` are for badges and text.

Do: one primary per view; pair a cloud button with cloud copy that says what it costs. Don't: a primary and a cloud button side by side competing; icon-only buttons without `aria-label`; cadmium text on a quiet button.
