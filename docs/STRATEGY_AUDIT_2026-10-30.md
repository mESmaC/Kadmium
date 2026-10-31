# Kadmium strategy audit — business plan and launch plan (30 October 2026)

**Documents audited:** [`BUSINESS_PLAN_AND_STANCE.md`](BUSINESS_PLAN_AND_STANCE.md) (v2.1) and [`MARKET_RESEARCH_AND_LAUNCH_PLAN.md`](MARKET_RESEARCH_AND_LAUNCH_PLAN.md), read together with the [R4 review](MARKET_RESEARCH_REVIEW_2026-10.md), the [numbers ledger](NUMBERS_2026-10.md), the [specsheet](SPECSHEET_AND_ROADMAP.md) and the [design-system voice guide](../design-system/1-voice.md).

**Lens.** This audit judges the plans against the *direction of the concept*: a local-first heterogeneous compute fabric (Windows, macOS, Linux, ARM) with polite yield, a desktop agent workspace (Condenser), elastic overflow to indexed cloud spot and to an output-metered community grid, an air-gapped office tier for regulated buyers, a campus and grant engine, and a USD-only steward-ownership charter. It does not re-audit what the code does today. The [R4 review](MARKET_RESEARCH_REVIEW_2026-10.md) and the open evidence review in PR #28 cover readiness, provenance and arithmetic; where those findings change the strategy they are cited, not repeated.

**Short verdict.** The concept is a genuine category, the pricing architecture is more rigorous than any adjacent competitor's, and the charter is a procurement feature rather than a slogan. The plans undercut that strength in three ways: they spread one founder across eight businesses in ninety days, they bury the one buyer who pays 90 percent gross margin under a manifesto aimed at hosts who earn twenty dollars a month, and they violate the project's own voice guide so thoroughly that the target community would discount the numbers. Section 4 lists eighteen angles the plans do not yet use. Section 9 gives the order of operations.

---

## 1. The audit in one page

| Area | Grade | Keep | Change |
| :--- | :---: | :--- | :--- |
| Category thesis (local-first + elastic overflow + ownership) | A | The four-quadrant position no competitor spans | Say it in one sentence and stop restating it |
| Pricing architecture (output-metered grid, index × spread, solvency rules, credit liability) | A− | All four mechanisms | Fix the included-credit inversion in the $29 plan; one credit unit, one multiplier |
| Software tiers (Lab, Office, MSP) | A | Margin profile and buyer fit | Make them the spine of the launch; price Office by node count |
| Supply Grid and host economics | B | Buyer-constraint-first clearing, no-demand-no-payout | Reposition hosts from "income" to "reciprocity and fleets"; defer the residential grid behind measured multi-node |
| Regulated SMB tier | B+ | 100 percent on-prem boundary, the FWHT honesty | Hero workload should be documents and retrieval on 2–3 office machines, not 70B; prepare for BAA requests anyway |
| Condenser as the Day-1 wedge | A− | In-product upgrade at OOM and 429 | Add step-level hybrid routing and a spend meter (§4.1) |
| Burst and CloudShield | B+ | Index-plus-spread, single-hop topology rule | Sell CloudShield as continuity, priced per protected node |
| Academic and grants | A− | STTR triple play, Scholar program | Target 4 March 2027; sell the campus fabric to central IT, not only to a professor |
| Crowdfunding and hardware | C | The ODM gate | Cut hardware and Kickstarter from the launch narrative; replace with OEM certification (§4.7) |
| Competitive landscape | B− | Teardown format | Add the DIY stack, FreeToken, colibri, GPUStack, vLLM/SGLang disaggregation, MLX distributed; mark Petals dormant |
| Civic narrative | B+ | PJM and IEA framing | It is the press hook, not the host-recruitment hook |
| Timeline and gates | C+ | Two-stage bridge shape | Milestone-gated, not date-gated; present the Day-90 base case software-only |
| Voice and brand | D | The design-system guide exists | The plans violate it on most pages; one product name, one workspace name, one marketplace name |

---

## 2. Steelman: the strongest form of the plan as written

A fair reader should be able to see why this plan could work before seeing what is wrong with it.

### 2.1 The thesis is a category, not a feature

Every adjacent product occupies one quadrant. Ollama and LM Studio run one machine or go fully to cloud. Exo pools machines but is Apple-centric and has no overflow. SaladCloud and the DePIN networks sell other people's GPUs but cannot shard across the buyer's own desk. Prime Intellect and RunPod aggregate datacenters and ignore local hardware. Kadmium is the only plan that describes, in one runtime, cross-vendor LAN pooling, a desktop agent workspace, sub-second overflow to indexed spot, and a USD-denominated output-metered grid. That is a defensible position because the pieces reinforce each other: the free local tier generates the demand signal, overflow is the natural upgrade moment, and the grid is where overflow eventually clears cheapest. The plan's comparison table in business plan §1.2 is the best single artifact in either document.

### 2.2 The pricing architecture solves the three problems that killed earlier P2P compute

1. **Output-metered clearing** prices what the buyer receives (tokens, verified rollouts, gigabytes seeded) rather than what the host owns. The plan's own worked examples show why capacity pricing fails: an RK3588 priced per gigabyte of RAM costs the buyer fifty times serverless. Buyer-constraint-first clearing is the correct insight and the plan derives host payout from it instead of the other way round.
2. **Index-plus-spread burst pricing** removes fixed-price inversion. Static retail tables lost money the moment wholesale moved; a daily-refreshed index with a published spread cannot.
3. **Solvency rules** (no demand, no payout; segregated 75 percent escrow; T+7 verified settlement; zero take-or-pay leases; credit redemption liability booked at issuance) are the kind of discipline that appears in a plan only after someone has watched a marketplace die of float.

Most competitors' plans have none of these. Kadmium's has all four, with the formulas written down.

### 2.3 Monetization is sequenced by margin, and the margin is where the engineering is easiest

The software tiers (Lab and Team at $199, Private Office at $499, MSP at $299 wholesale) carry 85–93 percent gross margin and require no WAN, no grid and no cloud account. They monetize the planner, RBAC, audit logging, the registry and the workspace. Burst carries a 20 percent spread and needs the hardest engineering. The plan correctly puts software first and burst second. Read generously, the two-stage bridge in business plan §8.1 is a plan to earn margin from what is nearest to done while the distributed kernels mature.

### 2.4 The honesty discipline is itself a go-to-market asset

The Measured versus Modelled labelling, the dated numbers ledger, the process invariant that every figure traces to one source, and the explicit rejection of orthogonal rotations as a compliance claim are rare in this market. The target community (r/LocalLLaMA, Hacker News, homelab) checks numbers and remembers who inflated them. A plan that volunteers "this cell is modelled" earns the right to be believed when it says "this cell is measured."

### 2.5 The charter is a procurement feature

Universities, government agencies and regulated SMBs cannot hold utility tokens or pass audits on anonymous nodes. A USD-only, steward-owned, no-token marketplace is not merely ethical; it is the only form of community compute those buyers are allowed to purchase. If SaladCloud's integration with the Render Network is as the plan reports it, the no-crypto consumer-grid position is vacant. The 75/20/5 split and the 4× median-wage compensation cap also pre-empt the "VC-backed extraction" objection that every DePIN project eventually faces.

### 2.6 The civic narrative is timely and factually anchored

Datacenter load driving PJM capacity prices, IEA's doubling projection, and residential tariffs rising while ratepayers receive none of the revenue is a real and current grievance. "Pay the people who already bought the silicon" is a sentence a journalist can quote. The plan grounds it in a hardware census rather than hand-waving.

### 2.7 The channel choices are the cheap ones

The MSP channel reaches law firms and clinics without an enterprise sales team. The STTR route gives a professor a six-figure subaward for benchmarking Kadmium on their own lab, which is simultaneously supply, credibility and non-dilutive capital. The Scholar program copies a mechanism (GitHub Student Pack) that is known to work. Creator seeding is priced at the realistic mid-tier rather than celebrity rates.

### 2.8 Condenser is the right Day-1 surface

A multi-provider agent workspace that already routes to Ollama, OpenAI and Anthropic gives users a daily-driver reason to install Kadmium before any distributed kernel ships. The in-product upgrade prompt at the two moments of real pain (an out-of-memory error, a 429 rate limit) is the correct conversion design.

### 2.9 Marketing can own concrete mechanisms

A $15 USB4 cable instead of a $600 switch. Layer 0 and Layer N−1 pinned locally so no prompt or output leaves the machine. A game launches and the layers move within 200 ms. Merkle-verified 64 KiB weight pages. These are demonstrable on camera and match the voice guide's "mechanism over adjective" rule.

---

## 3. Where the plan undercuts its own thesis

### 3.1 Scope sprawl: eight businesses, one founder, ninety days

Counted across both documents, the ninety-day plan commits to: five subscription tiers; three prepaid pack sizes; a six-tier Kickstarter including two hardware SKUs; a Reg CF equity round; an MSP partner program; a university STTR submission plus DOE, NAIRR, state and foundation grants; a Scholar program with campus ambassadors; a Sentinel server room sourced from donated hardware; a Phone Farm product line; Nintendo Switch node support; a four-loop referral engine with affiliate payouts; and the native multi-node kernel cutover. Each is individually reasonable. Together they guarantee that none is done well, and they make the plan read as a portfolio rather than a company.

**Recommendation.** Choose a spine and state it on page one: *the engineering team that wants private, rate-limit-free inference on the machines already in its office.* That buyer pays the Lab and Office tiers, is reachable through MSPs and professors, uses Condenser daily, and hits the overflow moment naturally. Everything else is either a flywheel that follows (Supply Grid, Scholar credits), a press hook (civic narrative), or a later option (hardware, Reg CF, phone farms).

### 3.2 Brand sprawl and voice violations

The documents use Kadmium, Kadmium Reactor, Cd-48, Control Rod, Condenser, Blades, CloudShield, Supply Grid, Grid-Zero, Kadmium Cloud, Kadmium Local and the Power Plant inference harness as if each were a product. The voice guide is explicit: the reactor metaphor is a tagline, not a theme; Cd-48 is a signature, never a name; no exclamation marks; no questions as headings; banned words include seamless, blazing, revolutionary, unleash and 10x. The launch plan has headings phrased as questions (§11.1–11.3), dozens of exclamation marks, and words such as "unassailable", "undefeated", "slam-dunks", "rocket fuel", "explosion", "unstoppable" and "unbeatable". The business plan is better but still calls a tier "Kadmium Reactor" in one document and "Supply Grid" in the other.

This matters commercially. The community the plan targets reads a marketing document the way it reads a benchmark: hyperbole lowers the prior on every number next to it.

**Recommended taxonomy.**

| Thing | Name | Never |
| :--- | :--- | :--- |
| The product and the binary | Kadmium / `kadmium` | Kadmium Local, Community Edition, Kadmium Reactor |
| The managed plane | Kadmium Cloud | KCloud, the cloud tier |
| The agent workspace | Condenser | Power Plant, Blades as a product (Blades is a feature of Condenser) |
| The marketplace | Supply Grid | Reactor, Reactor Credits, Grid-Zero (keep Grid-Zero only as a campaign title) |
| The failover feature | CloudShield | a product line |
| The signature | `Cd-48` in a footer or label | a heading or a hardware name |

Appendix A lists voice violations with rewrites.

### 3.3 The manifesto and the host table contradict each other

The launch plan opens with "pay the people who would otherwise suffer from scaling operations." The business plan's modelled host table then shows a Mac mini earning about $15 a month and a gaming desktop about $21 after electricity, at duty cycles the plan itself calls conservative. Both are honest. Together they tell a reader that the movement pays less than a streaming subscription.

The fix is positioning, not numbers. Residential host value is **reciprocity**: contributing overnight earns credits that let you run a model your desk cannot hold, which is worth far more to that user than the cash equivalent. Cash-grade earnings belong to **fleets**: a studio, campus lab or esports arena with two hundred machines on commercial power and symmetric fibre, where $20 per node per month is real money and the operator already pays the electricity. The civic narrative is the **press hook** for why the company exists; it should not be the recruitment pitch to a gamer. Lead the host story with "run bigger models on what you own, and let your hardware earn the overflow it needs"; lead the fleet story with revenue per rack; lead the press story with ratepayers.

### 3.4 Pricing design flaws that are strategic, not arithmetic

PR #28 documents the arithmetic (the $29 plan's $35 of credits cost $28 at index × 1.25; the give-$25-get-$25 loop loses on the first sale; the conversion ratios are mislabelled). The strategic version of those findings:

- **Included credits must not exceed the price.** A plan whose credits are worth more than its fee invites pure arbitrage and makes every active subscriber a loss. The rule should be: included credits at or below 60 percent of plan price, or credits redeemable only on community supply where the 75 percent host share applies.
- **One credit, one multiplier.** The plans issue 1.25× overnight credits, 2.0× Scholar credits, $100 VIP bonuses, $25 referral credits each side, $10 bench credits, $35 included credits and $500 lab credits. Each is a separate liability and a separate sybil vector (promo credits for compute are the most farmable promotion there is). Collapse to one unit, one multiplier, verified-device gating on any promotional grant, and hard monthly caps.
- **Office pricing by node count.** A flat $499 per office is generous to a 40-seat firm and expensive for a 4-seat one. Tier by nodes (up to 10, 25, 50) or price per node with a floor. Annual prepay at a discount is a cleaner working-capital source than a Reg CF round.
- **Burst spread.** Twenty percent gross on burst is thin for a business that also carries support. The deposit-backed 1.08× wholesale pass-through in launch plan §11.3 is a good retention device but should not be offered until the 1.25× rate has been shown to clear.

### 3.5 The hero number is the wrong comparison

The one measured figure in both plans is a 3.98× speedup over single-PC cold NVMe paging. Nobody chooses cold paging; the community's real alternative on the same two machines is llama.cpp RPC, Exo, or simply a smaller quantization on one box. A reader will ask "versus what?" and the honest answer undermines the headline. (PR #28 separately questions the run's provenance.) Whatever is measured next, choose the comparison the audience would have run themselves: Kadmium versus llama.cpp RPC on the identical mixed-OS pair, and an absolute tokens-per-second on a named model and named hardware.

### 3.6 The competitive map is missing the entrants that matter most

| Missing | Why it matters |
| :--- | :--- |
| The DIY stack (Tailscale + llama.cpp RPC, or Ollama on each box behind a reverse proxy) | This is what r/LocalLLaMA actually does today; it is the real incumbent for the free tier |
| FreeToken and colibri | Cited as inspirations; R4 notes they own the single-machine giant-MoE story and ship coding-agent APIs. The plan must say what Kadmium adds above one machine |
| GPUStack, vLLM and SGLang disaggregated serving, Ray Serve | The lab and team buyer's alternative for a homogeneous rack |
| Apple MLX distributed, two DGX Sparks over ConnectX, Strix Halo pairs | Vendor-native pooling for the two most desirable prosumer platforms |
| Petals | Listed as live; dormant since 2024 (R4 §12). Mark it |

The teardown's competitive claim should become: "Kadmium is the layer above one machine: pooling, polite yield, live shifting, privacy-tiered sharding, verification, credits, burst and training. For one machine, use FreeToken or colibri; Kadmium can drive them."

### 3.7 Hardware is a distraction at launch

The RK3588 node is correctly gated on a signed ODM quote and correctly described as a coordinator rather than a sidecar. It still carries a 14 percent modelled margin during a memory supercycle, an untested assumption about FCC treatment of a re-etched enclosure, fulfilment and RMA burden, and a Kickstarter campaign whose single launch day is the one thing a bootstrapped company cannot repeat. Section 4.7 proposes the zero-inventory alternative.

### 3.8 The regulated office tier is pitched at the wrong workload and under-prepared for the BAA conversation

- **Workload.** Launch plan §0.3 sells law firms and clinics on "70B+" models pooled from "16–64 GB" office machines. Most such offices run 8–16 GB laptops. The workload those buyers actually want is document summarization, extraction, drafting and retrieval over their own files, which a 7B–14B dense model or a 30B-class sparse MoE on two or three machines serves well. Pitch documents and retrieval; let 70B be the Lab tier's story.
- **BAA.** The plan reasons that on-prem software never touches PHI, so no Business Associate Agreement is needed. That is correct for the software and wrong for the sales process: any vendor with remote support access is a business associate, and every clinic's compliance officer will ask for a BAA regardless. Decide now between a no-remote-access support model and a BAA template plus cyber-liability cover ready by Gate 3. The same applies to the MSP, who will be the business associate in practice.
- **MSP enablement.** MSPs sell what their vendor's partner portal makes easy and boring: NFR licences, an RMM-deployable installer, a co-branded two-page "private AI for law firms" sheet, and a monthly recurring line they can mark up. The plan has the economics but not the enablement list.

### 3.9 Date-gated instead of milestone-gated

Gate 2 (days 36–65) contains the native multi-node kernel cutover and the Supply Grid beta, which is the entire hard engineering problem. Dates attached to that are a credibility cost with investors and grant reviewers, who will read them as a sign the author has not scoped the work. Gates should fire on milestones (a reproducible two-node measurement; a first paid office in production; a first fleet signed) and the Day-90 base case should be presented software-only, with burst and grid billings as upside. Section 6 restates it.

### 3.10 Crowdfunding before measured demos spends the one unrepeatable launch

A Kickstarter's algorithmic boost comes from the first four hours. Launching it in week three, before a head-to-head measurement exists, spends that moment on a modelled claim. Reg CF adds a Form C, a CPA review and ongoing disclosure to a one-person company. Sequence both behind Gate 2, and only if the Lab and Office tiers have not already produced the working capital (annual prepay and Founding Lab setup fees are 69 percent margin).

### 3.11 Grant timing

The NSF 26-510 deadline the plan names (4 November 2026) is five days from this audit. The Project Pitch can go in now; the full proposal should target 4 March 2027, and the PI employment rule (at least 51 percent at the small business) should be settled before a name is written down.

---

## 4. Angles the plan does not yet use

Ranked by fit with the concept and by cost to test. Each entry gives the angle, why it fits, the first move, and the falsification test.

### 4.1 Step-level hybrid routing: frontier for planning, local for doing

An agent task is fifteen to fifty tool turns. Most are mechanical (read, grep, edit, run tests, summarize a diff) and a small number need frontier reasoning. Condenser already routes to Anthropic, OpenAI and Ollama; the missing product is routing **by step**, not by model: frontier for the planning turn, a local model on the mesh for the mechanical turns. Add a per-task spend meter ("this task cost $4.12 on the API; 60 percent of its turns could have run on your desk for $0") and the upgrade prompt writes itself.

- **Why it fits.** It makes the free tier valuable to people who will never leave Claude Code or Cursor, it is the strongest answer to "cheap APIs are simpler", and it generates real agent-workload telemetry that calibrates the grid's pricing.
- **First move.** A routing policy in Condenser with three modes (frontier, local, hybrid) and the spend meter. Publish the hybrid savings on a public coding benchmark.
- **Falsifies if** hybrid routing degrades task success on SWE-bench-style evaluations by more than the cost saving justifies.

### 4.2 Rate-limit overflow as the Team headline

The plan mentions rate limits in passing. For a five-developer team on capped subscriptions, "no developer is ever blocked by a 5-hour limit" is a sharper headline than any dollar saving. Sell the Lab and Team tier as overflow for the tools the team already uses.

### 4.3 The sanctioned alternative to shadow AI

Companies that ban ChatGPT discover their staff use it anyway. An on-prem fabric is the CISO's sanctioned alternative. This reframes the Office tier for non-regulated SMBs and gives MSPs and MSSPs a security reason to sell it. First move: a one-page "shadow AI" sheet for MSPs, and an audit-log demo.

### 4.4 Demand response and carbon-aware compute

The plan uses the ratepayer grievance as a narrative. It can be a **partnership**. The power-price floor guard and time-of-use awareness already in the design make a Kadmium fleet a flexible load: it runs when power is cheap and backs off when the grid is tight. Utilities and ISOs pay for that flexibility, and DOE, ARPA-E and state energy offices fund it. Publish tokens per kilowatt-hour and the off-peak share of grid work as live public metrics.

- **First move.** One utility or community-choice aggregator pilot in a PJM state; a DOE SBIR framed as grid-aware load shifting (already sketched in launch plan §13.3).
- **Falsifies if** demand-response revenue per node is immaterial relative to compute revenue and the integration cost exceeds it.

### 4.5 Sovereign and regional compute

A pure-Rust, air-gapped, no-US-cloud fabric fits EU public-sector and SMB buyers under GDPR and the AI Act, and similar sovereignty programmes in Canada, India and the Gulf. European electricity makes residential hosting unattractive and makes **buyers** more motivated, which suits the software tiers. First move: EUR pricing, a GDPR data-flow diagram, and one EU MSP partner.

### 4.6 Fleets, not gamers, as the supply anchor

The plan names VFX studios and esports arenas. The larger pools are PC bangs in Korea, net cafés across China and South-East Asia, university computer labs, school-district refresh cycles, and post-mining GPU hosts that still hold power contracts and racks of consumer cards (no token involved; they want USD work). Five fleets of two hundred machines on commercial power beat five thousand gamers on residential tariffs, and they can sign an SLA bond. First move: two signed fleets before any residential host campaign.

### 4.7 OEM certification instead of own hardware

Replace the RK3588 Kickstarter with a "Kadmium-ready" programme: Framework, Minisforum, Beelink, Radxa, Raspberry Pi and System76 ship the image or a pre-configured profile; Kadmium publishes the certification matrix and takes an affiliate margin. Zero inventory, zero FCC exposure, co-marketing from partners who already have an audience. The reference images for Pi 5, Rock 5B and Zimaboard that the plan already promises are the first rung.

### 4.8 Verified weights as a trust product

The `.ktb` format carries Merkle roots. Signed official builds of open-weight models (DeepSeek, Qwen, Mistral, gpt-oss) with published roots, plus Verde verification of execution, is a "verified weights, verified execution" story that no serverless provider offers. Partner with Hugging Face as a `.ktb` export target. This is also the natural home for the model-pack royalty loop in launch plan §20.4.

### 4.9 The RL environment marketplace

Loop 4 (royalties for `.ktb` packs and Dream-RSI worlds) is the seed of a real product: a registry of deterministic, fuel-metered WASI verification environments with royalties to authors. The competitor is Prime Intellect's environments hub; the differentiator is deterministic verification on commodity CPUs and a USD royalty. First move: ten environments ported, one paying RL lab.

### 4.10 Own the benchmark

The "will it run on my desk" calculator is good; a public, reproducible **heterogeneous mesh leaderboard** with user-submitted `kadmium bench --share` results is better, and head-to-heads against llama.cpp RPC and Exo on identical mixed hardware are the content the community will share. Whoever owns the benchmark owns the category definition.

### 4.11 A public "no new datacenters" counter

Turn the manifesto into a dashboard: GPU-hours served without a new facility, megawatt-hours shifted off-peak, dollars paid to hosts, grants funded from the 5 percent pool, and (always) zero litres of cooling water and zero kilograms of concrete. Journalists quote counters.

### 4.12 CloudShield as business continuity

Sell failover as insurance, priced per protected node, to MSPs who already sell backup and disaster recovery. "AI continuity" sits next to "backup" on their price list and is a recurring line they understand.

### 4.13 Hardware-tribe marketing

Each hardware community has its own forum and its own objection: Mac users (Exo territory), NVIDIA gamers (polite yield), Strix Halo and Framework buyers (128 GB UMA pairing), Pi and SBC homelabbers (coordinator and seeder roles), Switch modders and phone-farm builders (e-waste nodes). One hero demo per tribe, posted in that tribe's venue, beats one generic launch video.

### 4.14 Formalize the charter

A steward-ownership structure with a golden share, a published charter and a fiscal sponsor or foundation for the grant pool makes Kadmium eligible for Mozilla, Sloan, NLnet and similar public-interest funding, and converts "we promise" into "it is in the bylaws". It also answers the enterprise question "what happens if you are acquired".

### 4.15 MCP-first developer distribution

`kadmium-mcp` in the MCP registries, a base-URL drop-in for Cline, Continue and OpenCode, and Condenser's plugin interface make Kadmium the compute layer *inside* other agent tools. "Kadmium inside" reaches developers who will never open a new workspace.

### 4.16 Campus fabric sold to central IT

The STTR route reaches one professor. Central IT owns the five thousand lab machines. Sell the campus fabric as an IT product with the Scholar programme as its demand side, and position it as a NAIRR-compatible resource for non-R1 institutions and community colleges.

### 4.17 Supply-chain collateral for defence buyers

For CMMC subcontractors the pure-Rust story becomes concrete with an SBOM, `cargo vet` or `cargo crev` attestations, reproducible builds and a signed release process. Produce the artefacts once; they are sales collateral for every regulated buyer and a partner asset for CMMC registered practitioner organizations.

### 4.18 Measurement as content

Every `kadmium bench` run on a new mesh is a post. A standing weekly measurement series ("this week: RTX 5070 + M4 mini over USB4") is cheaper than ads, compounds in search, and is the honest version of the creator-seeding plan.

---

## 5. A tighter narrative

**One line.** Every machine you own, as one reactor. *(From the voice guide; the only place the metaphor appears.)*

**Positioning statement.** Kadmium pools the GPUs, Macs, laptops and NAS you already have into one inference, training and RL fabric; overflows to indexed cloud spot or to a USD-metered community grid only when a job needs more than you own; and never sends a prompt or an output off your hardware.

**Three pillars, in the order the buyer meets them.**

1. **Pool.** Windows, macOS, Linux and ARM over Ethernet or a $15 USB4 cable. Free to self-host, forever. MIT or Apache-2.0.
2. **Overflow.** Burst at the wholesale index plus a published spread, or clear batch work on the Supply Grid at a discount to serverless. Layer 0 and Layer N−1 stay on your machine.
3. **Keep.** Your hardware yields to you in under 200 ms. Your office runs air-gapped. Seventy-five percent of every grid dollar goes to the host. No tokens, no telemetry.

**Per-buyer lines.**

| Buyer | Line |
| :--- | :--- |
| Engineering team | No developer is ever blocked by a rate limit, and no source code leaves the office |
| University lab | Your idle lab machines become the cluster your grant could not buy |
| Law firm, clinic, defence shop | Private AI on the PCs you already own, inside your own firewall |
| RL and post-training lab | Verified rollouts on commodity CPUs at a discount to serverless, with a receipt |
| Fleet operator | Your overnight idle machines earn USD, with an SLA bond and a T+7 settlement |
| Press | We don't need any additional datacenters |

**Proof points, in the order they will exist.** A two-node measured head-to-head; the USB4 ring on camera; the Wireshark privacy capture; the office installer running on three laptops; the first fleet's monthly statement.

**Words to retire.** Reactor as a product name, Grid-Zero as a product name, Control Rod, unassailable, undefeated, slam-dunk, rocket fuel, explosion, unstoppable, unbeatable, every exclamation mark.

---

## 6. A milestone-gated commercial sequence

| Gate | Fires when | Commercial motion | What is sold | Not yet claimed |
| :--- | :--- | :--- | :--- | :--- |
| 0 | A reproducible two-node measurement against llama.cpp RPC exists | Measurement post in each hardware tribe; calculator live; design-partner applications | Nothing; Founding Lab setup fees may be taken as deposits | Any tok/s that is not in the measurement |
| 1 | Condenser hybrid routing and the spend meter ship | Team and Lab tier on; annual prepay; MSP recruiting begins | Lab and Team; Founding Lab setup | Burst, grid, failover |
| 2 | The office installer runs unattended on three mixed laptops with audit log and SSO | Private Office direct and via two MSPs; shadow-AI and continuity sheets | Office; MSP wholesale | Burst, grid |
| 3 | Index-plus-spread burst clears for thirty days at or above the published spread | CloudShield and prepaid packs on; the $29 plan with credits at or below 60 percent of price | Burst; CloudShield per node | Residential grid |
| 4 | Two fleets signed with SLA bonds and T+7 settlement running | Supply Grid for batch and WASI work; environment registry; credits for residential hosts | Grid capacity to labs; royalties | Hardware |
| 5 | Everything above is in production and cash-positive | OEM certification; grant-funded campus fabrics; optional hardware only with a signed ODM quote | Certification; campus contracts | Kickstarter unless the above is funded another way |

**Day-90 base case restated software-only** (same unit counts as business plan §4.2, no burst, no packs):

| Line | Units | Monthly |
| :--- | :---: | :---: |
| Lab and Team at $199 | 35 | $6,965 |
| Office at $499 and MSP at $299 (blended $399) | 12 | $4,788 |
| Software subtotal | | $11,753 |
| Burst, grid and packs | | upside, reported separately |

Fixed infrastructure at $750 a month is covered by the second office. This is the number to budget against; the $35,903 figure stays as the stretch case.

---

## 7. Scorecard by section

**Business plan**

| Section | Verdict | Note |
| :--- | :---: | :--- |
| §1 Thesis and landscape | Keep | Trim to one page; the comparison table is the asset |
| §2 Pay-in tiers and index | Sharpen | Fix included credits; price Office by nodes; add node-count tiers |
| §3 Pay-out clearing | Keep | Add the fleet-first supply strategy |
| §4 Liability and 90-day model | Sharpen | Software-only base; relabel conversion ratios (PR #28) |
| §5 Hardware economics | Defer | Move to an appendix; replace with OEM certification |
| §6 Security and compliance | Keep | Add BAA stance and support-access model |
| §7 Charter | Keep | Formalize (§4.14) |
| §8 Launch architecture and gates | Rewrite | Milestone-gated (§6) |
| §9 Unit economics and treasury | Keep | Add loaded CAC including founder time |

**Launch plan**

| Section | Verdict | Note |
| :--- | :---: | :--- |
| §0 Macro problems and positioning | Sharpen | Press framing; not the host pitch |
| §1 Diagnosis and wedge | Keep | Best short statement of the strategy |
| §2 Competitive teardown | Sharpen | Add missing entrants (§3.6) |
| §3 Reality checks | Keep | The most credible writing in either document |
| §4 Pricing | Sharpen | As business plan §2 |
| §5 Thirty-day playbook | Rewrite | Behind Gate 0; outbound to fleets and MSPs first |
| §6 Revenue model | Sharpen | Software-only base |
| §7 Regulated verticals and MSP | Sharpen | Document workload; BAA; enablement kit |
| §8 Coding-agent moat | Keep | Add hybrid routing and the spend meter |
| §9 Spot landscape | Keep | Mark unverified rates |
| §10 FinCEN and tax | Keep | Counsel review before launch |
| §11 Awareness, trust, staking | Sharpen | Rename headings; keep the trust stack and SLA bond |
| §12 Crowdfunding | Defer | Behind Gate 2 or drop |
| §13 Academic and grants | Keep | March 2027; central IT |
| §14 Master timeline | Rewrite | As §6 |
| §15 Swarming and striping | Keep | Already scoped honestly to throughput work |
| §16 Advertising | Defer | After retention is observed; measurement series first |
| §17 Scripts and copy | Rewrite | Voice guide (Appendix A) |
| §18 VIP funnel and BOM | Defer | With §12 |
| §19 Grant playbook | Keep | Strong as written |
| §20 Referral loops | Sharpen | One credit unit; caps; verified devices |
| §21 Sentinel server room | Keep, shrink | Three to four donated boxes for referees and seeding; not a programme |
| §22 Phone farms and Switch | Defer | Tribe content, not launch scope |
| §23 Late-October multipliers | Keep | Condenser and USB4 are the Day-1 story |

---

## 8. Metrics and falsification tests

| Hypothesis in the plan | Metric | Fails if |
| :--- | :--- | :--- |
| Free local pooling creates demand for overflow | Share of active clusters that hit an OOM or 429 prompt per month | Under 10 percent after three months |
| Teams pay for rate-limit relief and privacy | Lab and Team conversions from design-partner calls | Fewer than one in ten qualified calls converts |
| Offices buy documents-and-retrieval on their own machines | Office pilots that renew after month one | Under half renew |
| MSPs will resell | Offices deployed per signed MSP in ninety days | Fewer than two |
| Index-plus-spread clears | Realized spread on burst, thirty-day trailing | Below 15 percent or negative in any week |
| Fleets supply SLA-grade capacity | Fleet nodes with bond posted and T+7 settled | Fewer than two fleets by Gate 4 |
| Residential hosts value credits over cash | Credit-track versus cash-track opt-in, and redemption rate | Redemption above 70 percent (liability model breaks) or opt-in under 20 percent |
| Hybrid routing saves money without losing quality | Spend reduction and task success on a public coding benchmark | Success drops more than two points per 30 percent saved |
| The civic narrative earns press | Earned placements per quarter in named outlets | Zero in two quarters |

---

## 9. Order of operations

1. Adopt the spine (engineering teams; Lab and Office) and write it on page one of both plans.
2. Apply the brand taxonomy and the voice guide to both documents; remove every exclamation mark and question heading.
3. Replace the date-gated timeline with the milestone gates in §6; restate the Day-90 base case software-only.
4. Fix the credit design: one unit, one multiplier, included credits at or below 60 percent of plan price, caps and verified-device gating on promotions.
5. Submit the NSF Project Pitch now; plan the full proposal for 4 March 2027.
6. Choose the hero measurement: Kadmium versus llama.cpp RPC on one mixed-OS pair, plus an absolute tokens-per-second on a named model.
7. Build the Condenser spend meter and hybrid routing; publish the savings.
8. Sign two fleets before any residential host campaign.
9. Reposition the Office tier on documents and retrieval; write the BAA stance and the MSP enablement kit.
10. Cut hardware and Kickstarter from the launch narrative; open OEM certification conversations.
11. Add the missing competitors to the teardown and mark Petals dormant.
12. Stand up the public counter and the mesh leaderboard as the standing content programme.

---

## Appendix A — Voice-guide violations with rewrites

| Location | As written | Rewrite |
| :--- | :--- | :--- |
| Launch plan, opening note | "The Undefeated Core Thesis … Why This Positioning Is Unassailable" | "The thesis. Compute buyers can pay the people who already own the silicon and already pay the electric bill." |
| Launch plan §11.1 heading | "How Do We Get People to Know We Exist? (The 5-Channel Awareness Engine)" | "Awareness: four channels" |
| Launch plan §11.2 heading | "How Do We Get People to Trust What We Offer?" | "Trust: four verifiable pillars" |
| Launch plan §13.2 | "turns our grant applications into slam-dunks" | "strengthens the Broader Impacts section of every grant application" |
| Launch plan §16.2 heading | "How to Guarantee a Day-1 Kickstarter Explosion" | "Pre-launch reservations" |
| Launch plan §22.1 | "Grant Rocket Fuel" | "Why this supports grant applications" |
| Launch plan §3.1 | "transferring 33 MB takes 8–13 seconds per hop!" | "transferring 33 MB takes 8–13 seconds per hop." |
| Launch plan §9 | "locking in Kadmium's 20.0% retail gross spread (Index × 1.25)!" | "which preserves the published spread." |
| Launch plan §17.3 | "we're dropping $25 in free middle-layer Burst Credits to everyone who posts…" | "Post a multi-node benchmark or a bug report in this thread and we will add $25 of burst credit to your account (one per verified device)." |
| Both plans | "Kadmium Reactor", "Reactor Credits", "Grid-Zero AI" as product names | "Kadmium", "Supply Grid credits", and "Grid-Zero" only as a campaign title |
| Both plans | `Cd-48` in headings and as a hardware product name | `Cd-48` as a footer signature only |
| Launch plan §0.1 | "turn dormant silicon into direct USD payouts" (aimed at individuals) | "earn the overflow your own hardware needs" (individuals); "earn USD per rack with an SLA bond" (fleets) |
