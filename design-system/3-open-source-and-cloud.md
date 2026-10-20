# Open source and Kadmium Cloud

Kadmium is open-core in the Supabase / Tailscale shape: the whole fabric is free to self-host forever, and Kadmium Cloud is a paid managed plane that wraps the hardware a person already runs. The identity makes that split visible on every screen without ever making the free product look like a trial.

## The colour code

| | Self-hosted Kadmium | Kadmium Cloud |
| --- | --- | --- |
| Fill | `cadmium` + `on-cadmium`, bordered `cadmium-edge` | `cherenkov` + `on-cherenkov` |
| Text, icon, border | `cadmium-text` | `cherenkov-text` |
| Tint | `cadmium-soft` | `cherenkov-soft` |
| Badge | `local` | `cloud` |
| Button | `variant="primary"` | `variant="cloud"` |
| Plan card | `product="self-hosted"` | `product="cloud"` |
| Rods and meters | cadmium | cherenkov |
| What it marks | hardware the person owns; the binary; the open-source product | burst GPUs, spot instances, the managed control plane, the failover vault, billing |

The code is strict in both directions. A local node is never blue, even when it is the one being rebalanced; a rented A100 is never yellow, even when it is doing the most work. A view with no cloud involvement contains no cherenkov at all — a self-hosted dashboard is yellow, graphite and status, nothing more.

## Writing the two products

- The free product is simply **Kadmium**. Describe it by what it is: "One Rust binary. Free to self-host, forever. MIT or Apache-2.0." Never "community edition", "free tier", "lite" or "basic".
- **Kadmium Cloud** is described by what it *adds* to a cluster the person already has: burst GPUs attached in minutes, multi-provider spot arbitrage at a 60–85 % discount to a monolithic cluster, a pre-warmed `.ktb` vault that absorbs shards within milliseconds when the lab loses power and fails back to local hardware at $0 / hour when it recovers.
- Pricing is a number with a unit and a qualifier: "$0.89 / GPU-hour, from", "usage billing, no minimum". The self-hosted plan card says "$0 forever", set in `metric` like any other price — it is not styled as a discount.
- Upgrade moments are offered where the cloud is useful (a node went offline; a training run would finish sooner with one more GPU) in `body` copy with a `cloud` button, never as a banner, modal or lock icon on a feature.

## Plan cards

Use `PlanCard` for both, side by side, same size, free first. Each has a mono eyebrow with a rod bullet in its product colour, the name in `headline`, the price in `metric`, one-sentence summary, four or five features with rod bullets, and one full-width button (`primary` for self-hosted, `cloud` for cloud). No "most popular" ribbons, no greyed-out features on the free card.

## Mixed clusters

The common case is both at once: a local mesh plus a burst GPU or a failover vault. In the topology view and the `ShardRail`, cloud nodes sit last in flow order and are the only cherenkov elements on the canvas; in a `NodeCard`, `placement="cloud"` sets the badge, the icon tint and the meter colour together. During a handoff the moving shard keeps the colour of its *destination* once the boundary transfer completes — a shard that fails over to the vault turns blue when it lands there, and yellow again when it fails back.
