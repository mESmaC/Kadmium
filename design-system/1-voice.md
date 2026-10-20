# Voice and copy

Kadmium speaks like the engineer who built it: direct, specific, a little dry, certain of the physics and unimpressed by the industry. The README's own line is the register to match — *"The modern AI industry promotes a convenient myth: frontier model training requires hundreds of thousands of dollars in homogeneous datacenter GPUs."* State the myth, then the mechanism that breaks it.

## Names and casing

- **Kadmium** is the product and the company brand; capitalise it like a proper noun in prose and headings. The domain is written `kadmium.ai`.
- **`kadmium`** (lowercase, mono) is the binary, the CLI and the crate prefix: `kadmium serve --port 7848`, `kadmium-sharding`, `.ktb` files. In code and commands, never "Kadmium".
- **Kadmium Cloud** is the managed product — two words, both capitalised, never "KCloud" or "the cloud tier".
- **Cd‑48** is a signature, not a name. Set it in `label` or `code` style beside the wordmark or in a footer; never as a heading, never explain it in the UI.
- Write real hardware the way its owner would: "M4 Mac Mini", "gaming laptop", "RTX 5070", "Raspberry Pi 5", "the NAS". Models by their canonical id in mono: `deepseek-v3-671b-moe`, `llama-3-70b-lora`.
- Methods keep their published names, capitalised as their authors do: DiLoCo, DeMo, DisTrO, SWARM, SkipPipe, RepOps, Verde, GRPO. Expand an acronym on first use in docs, never in the dashboard.

## Rules

- **Mechanism over adjective.** "Streams cold experts from NVMe at 7 GB/s" beats "lightning-fast expert streaming". Banned: seamless, blazing, powerful, revolutionary, effortless, magic, supercharge, unleash, 10x.
- **Numbers carry units and precision that is true.** "1,000×–3,000× less sync bandwidth", "46 / 64 GB", "a 30-second preemption notice". Round for prose, never for the dashboard.
- **You own the hardware; Kadmium does the work.** Address the operator as *you*; Kadmium is *it*. "Kadmium shifts those shards onto your Mac Mini" — not "we" and not "our platform".
- **Verbs first in the interface.** Buttons: "Start node", "Attach burst GPU", "Shard plan", "Evict". Sentence case everywhere; title case only for product names.
- **Status words are the vocabulary of the fabric** and are used identically in badges, logs and docs: resident, training, prefetching, joining, yielded (a node that stepped back for its owner — gaming mode, battery, thermals), throttled, offline, faulted.
- **The reactor metaphor is a tagline, not a theme.** One line on the cover or the home page — *Every machine you own, as one reactor.* / *Regulate the reaction.* — then plain engineering. No "core meltdown" errors, no "radiation" loaders, no atom icons.
- **Open source is stated, not sold.** "Free to self-host, forever. MIT or Apache-2.0, your choice." Kadmium Cloud is described by what it adds, in the same tone — never by implying the self-hosted product is lesser.
- **No emoji** in the product, the site or the docs. (The GitHub README may keep its section emoji as a GitHub convention; nothing else inherits them.)
- **No exclamation marks**, and no questions as headings.

## Examples

| Instead of | Write |
| --- | --- |
| Blazing-fast distributed inference on any hardware! | Asymmetric sharding across the GPUs, Macs and laptops you already own. |
| Our revolutionary cloud seamlessly scales your cluster | Kadmium Cloud attaches a burst A100 to your cluster and hands shards back when the lease ends. |
| Oops! Something went wrong 😬 | Node `gaming-laptop` faulted during layer 17 — refereed bisection pinned the fault to its CUDA reduction. Rerouting. |
| Memory usage: high | VRAM 11.8 / 12 GB — streaming cold experts from NVMe |
| Upgrade now to unlock the power of the cloud | Add a failover shield: a pre-warmed `.ktb` vault absorbs your shards within milliseconds if the lab loses power. From $0.89 / GPU-hour. |
