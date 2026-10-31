# Open source and Kadmium Cloud

Kadmium follows a proposed open-core shape: local planners, interfaces, and reference workloads are self-hostable, while Kadmium Cloud is a **planned** paid managed plane around a person's hardware. The identity makes that distinction visible without presenting simulated cloud features as live services or making the self-hosted product look like a trial.

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
- **Kadmium Cloud** copy describes target behavior in future tense: attaching burst GPUs, arbitraging approved provider offers, and failing over through a pre-warmed vault. Do not publish attachment times, discount ranges, RTOs, or automatic failback as current guarantees. Fixed CLI/API scenarios are labelled simulations.
- Pricing is published only after approval, with a number, unit, and qualifier. Do not reuse sample prices as live offers. The self-hosted plan card says "$0 forever", set in `metric` like any other price — it is not styled as a discount.
- Upgrade moments are offered where the cloud is useful (a node went offline; a training run would finish sooner with one more GPU) in `body` copy with a `cloud` button, never as a banner, modal or lock icon on a feature.

## Plan cards

Use `PlanCard` for both, side by side, same size, free first. Each has a mono eyebrow with a rod bullet in its product colour, the name in `headline`, the price in `metric`, one-sentence summary, four or five features with rod bullets, and one full-width button (`primary` for self-hosted, `cloud` for cloud). No "most popular" ribbons, no greyed-out features on the free card.

## Mixed clusters and zero additional datacenters

The target mixed-cluster case is a local mesh plus a burst GPU or failover vault. In topology mockups and the `ShardRail`, cloud nodes sit last in flow order and are the only cherenkov elements on the canvas; in a `NodeCard`, `placement="cloud"` sets the badge, icon tint, and meter colour together. During a simulated handoff, the moving shard takes its *destination* colour only after the scenario's boundary transfer completes. Label the state as simulated unless a live provider/session is actually attached.

This visual split also encodes Kadmium's grid positioning — **"We don't need any additional datacenters"**:
1. **Local silicon first (`--cadmium`):** prioritize existing consumer and lab hardware. Current `kadmium lease` prints an illustrative quote; it does not pay USD or issue redeemable credits. Polite-yield latency is a policy target/estimate, not a measured guarantee.
2. **Existing datacenter spot surplus (`--cherenkov`):** the target is to use existing spot capacity when local resources are insufficient. DeMo's ratio depends on tensor size and `top_k`; no fixed compression factor or datacenter-avoidance outcome is established. Environmental claims require a separate lifecycle analysis.
