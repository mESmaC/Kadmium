# NodeCard

One machine in the fabric: name, hardware line, placement badge, memory-tier meters, live status and the shard range it holds. The topology and Nodes views are grids of these, `space-4` apart.

You provide `name` as the person named the machine, a lowercase dot-separated `hardware` line, `placement` (`local` or `cloud` — it sets the badge, the icon tint and the default meter colour together), `kind` for the icon (desktop, laptop, compact, sbc, nas, gpu, cloud), `status` from the fabric's vocabulary, `meters` and a mono `shards` range. Put anything else (a link, a transfer in flight) in `children`.

Do: order cards in flow order — coordinator, pipeline stages, cloud nodes last. Don't: colour the card itself by status (the badge does that), add a left border, or show more than three meters.
