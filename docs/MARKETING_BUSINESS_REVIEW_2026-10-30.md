# Marketing and business review — 30 October 2026

**Decision:** pursue a small, paid, customer-owned compute pilot before launching a compute marketplace. Kadmium's strongest opportunity is making mixed local hardware useful and manageable for a repeatable workload. Its current evidence does not support the previous immediate-revenue, compliance, failover, or host-earnings promises.

This review steelmans the opportunity and tests the objections. It is desk research and repository inspection, **not customer discovery, a security audit, a legal opinion, or an independent hardware benchmark**. No customer interviews, revenue records, bank balances, signed partnerships, or incorporation documents were available. Recommendations, prices, budgets, and thresholds are proposals for review, not commitments or measured demand.

## 1. Scope and evidence rules

Repository baseline: [2e2040b5d8feae4a84da6fd306c0452dafff369e](https://github.com/mESmaC/Kadmium/tree/2e2040b5d8feae4a84da6fd306c0452dafff369e). Reviewed the business and launch plans, R4 research, numbers and benchmark ledgers, README, embedded marketing page, Condenser guide/provider boundary, CLI commands, spot provisioner, and clearinghouse implementation.

| ID | Repository evidence | What it establishes |
| --- | --- | --- |
| E1 | [Previous business plan](https://github.com/mESmaC/Kadmium/blob/2e2040b5d8feae4a84da6fd306c0452dafff369e/docs/BUSINESS_PLAN_AND_STANCE.md), especially §§1–4, 8–9 | Prior positioning, prices, benchmark interpretation, financial cases, and solvency claims |
| E2 | [Previous launch plan](https://github.com/mESmaC/Kadmium/blob/2e2040b5d8feae4a84da6fd306c0452dafff369e/docs/MARKET_RESEARCH_AND_LAUNCH_PLAN.md), especially §§11, 16, 18, 20, 23 | Paid channels, referral liabilities, hardware offers, and Day-1 conversion narrative |
| E3 | [Benchmark ledger](BENCHMARKS_2026-10.md), Run 1 §§1.1–1.7; Runs 2–3 | A reported scratch-harness run with unresolved provenance/configuration checks; separate reference-kernel loopback measurements |
| E4 | [Condenser guide](guides/condenser-workspace.md) and [provider boundary](../crates/kadmium-condenser/src/provider/mod.rs) | External providers can generate; Kadmium selection remains reference diagnostics; compute inspection previews placement |
| E5 | [Spot provisioner](../crates/kadmium-sharding/src/spot_provisioner.rs), query_live_orderbook / provision_optimal_middle_shard | A literal quote vector and returned container descriptor, not evidence of live procurement or provider API provisioning |
| E6 | [Clearinghouse](../crates/kadmium-core/src/reactor_clearinghouse.rs), evaluate_clearinghouse_ledger | Explicit deterministic settlement simulation with example buyers; not bank escrow, payment reconciliation, or real traction |
| E7 | [Embedded marketing page](../crates/kadmium-gateway/src/marketing_site.html), pricing, hero and supply calculator | Conflicting prices and promises remain in a customer-facing asset; this research PR does not change the application |
| E8 | [API contract](api-reference.md), [CLI](../crates/kadmium-cli/src/main.rs), [Cargo manifest](../Cargo.toml) | Execution/command boundaries and declared license metadata; metadata alone is not a complete release/legal package |

Use **observed** for repository content or a source's statement, **reported** for vendor/run-author results, **derived** for arithmetic, and **hypothesis** for proposed demand, performance, conversion, or economics. A dated constant is not automatically a verified market price. A vendor claim is not our benchmark. Recheck prices when quoting; record effective date, priority, region, availability, billing unit, and included costs.

## 2. Findings that change the commercial decision

| Priority | Finding and evidence | Consequence and correction |
| --- | --- | --- |
| P0 | E1 §8 and E2 §§11/17 describe Run 1 as Q4_K_M, 19.6 GiB, 16 GB PC, layers 0–15/16–39, wired 1GbE, zero swap. E3 records BF16, 71.90 GB, 32 GB PC, layers 0–24/25–39, CPU execution and JSON/HTTP with Wi-Fi variance. The harness was outside main; Mac memory/paging remains unresolved. | The advertised 3.98× result is not a reproducible product speedup. It compares a reported Rust path with cold single-PC paging, not an optimized equal-condition baseline. Withdraw the headline; engineering must commit provenance and reproduce matched baselines. |
| P0 | E4/E8 distinguish provider-backed generation from Kadmium diagnostics. E5 returns fixtures; E6 simulates settlement. | Module names and test counts do not establish Day-1 CloudShield, spot procurement or financial settlement. Sell a bounded evaluation using verified existing paths; product/engineering gate managed features separately. |
| P0 | E7 advertises $29 including $40 credits, $199 including $250 credits, +$1.15/hour hosting, compliant office AI, and sub-second failover. | **Publication blocker:** marketing/engineering must replace prices, calculator outputs, unsupported compatibility/compliance claims and CTA destinations before directing traffic to /site. These disagree even with the previous ledger. |
| P0 | A 1.25× wholesale markup implies 80% fulfillment cost and 20% gross margin. E1 reserves 75% for wholesale packs. A $29 plan with $35 fully used costs $28 before fees. | At the illustrative US card rate, −$0.14 remains before service costs. Host credits move obligations rather than extinguish them. Finance should separate subscriptions from compute and reserve against downside redemption. [S10–S11] |
| P0 | E1's “accrual margin” starts from prepaid cash sales and deducts a fulfillment reserve. | A reserve is not revenue recognition. Unconsumed packs are liabilities; consumed packs must not also count as fresh burst cash. Rebuild revenue, cash, balances and costs separately. |
| P1 | E1/E2 label 45/1,500 and 150/2,500 conversions as 1.5% and 3%; the ratios are **3% and 6%**. 520/6,000 is about 8.7%. | The model understates conversion difficulty. No measured funnel supports these cases. Replace adoption budgeting with a named-account experiment and delivery-capacity limits. |
| P1 | E2 offers $25 to each side after a $25 purchase: $75 total compute obligations costs $60 at full redemption. | The first sale loses **$36.03 after one card fee**, before service. A 15% affiliate fee also turns the $379 hardware offer's $54.73 contribution into **−$2.12**, before further shipping/tax variance. Reward retained software customers from contribution, with a fixed cap. |
| P1 | E1/E2 treat Salad's Render integration as abandoning USD buyers and Exo as Mac-only. | Salad still accepts cards; Exo lists Linux installs and depicts DGX Spark. Compare whole buyer alternatives, not an allegedly empty category. [S3–S5] |
| P1 | E1 calls the charter established, T+7 risk-eliminating and on-premise placement compliance-assuring. E2 promises zero FCC risk and tax/ESG donation benefits. | Simulations cannot establish legal structure, certification, tax status or payment finality. Label policies as proposals; substantiate each contractual claim before selling regulated use/hardware. [S11, S14–S17] |
| P1 | Steam concurrency × survey hardware percentages is presented as available GPU supply. | Denominators differ: logged-in people are not a measured fleet of idle, consented, compatible hosts. Supply is not demand. Remove this from TAM, utilization and financing arguments. |
| P1 | Zero-CAC, guaranteed viral conversions and $0-cost scholar credits underpin E2. | Founder time, support and fulfillment remain costs. Track loaded CAC, activation, renewal and contribution; suspend broad ads/crowdfunding until retention is observed. |
| P2 | Overlapping brands, tiers, workloads and buyer classes compete for attention. | Use Kadmium as the product, Condenser as its workspace and one first paid offer. Keep Reactor/Grid-Zero as optional mission language. |

The revised plans replace the previous commercial recommendations. Historical R4 and runtime fixtures remain, with scope notices. This **does not** repair the marketing page, simulated billing, supply pricing, or every roadmap statement elsewhere.

## 3. The strongest case for Kadmium

### A narrow entry point can support the larger vision

The valuable asset is not an inventory of supposedly free GPUs. It is the ability to convert a team's unevenly available machines into a dependable workflow with an understandable compute path. A small engineering/evaluation team may already own hardware, run open models and lose hours to setup, placement, reproducibility and deciding when cloud execution is justified. A fixed fee can be rational if a pilot demonstrates saved operator time or enables a needed workload within real constraints.

The repository offers credible building blocks: inspectable planning, worker transport, Condenser's provider adapters and bounded runs, and a measurement-oriented architecture. A supported workflow over an existing inference backend can test value while native distributed execution develops. If that workflow repeatedly exposes a need for mixed-device capacity, runtime development gains a paying design partner.

The differentiation to earn is **a tested mixed-OS/mixed-vendor support matrix, reproducible workload receipts, predictable scheduling and straightforward operation**. Free local use, clear data paths, fair host economics and an open mission can reinforce trust. They cannot substitute for execution, quality or reliability.

### Why this might still fail

| Strong objection | Best honest response | Evidence that would falsify the plan |
| --- | --- | --- |
| Cheap APIs or one larger machine are simpler. | Often true. Compare quality-matched tasks, admin time, latency, power and amortized hardware. | Ten qualified pilots find no material benefit over a properly configured single node/API. |
| Exo, GPUStack or LM Studio already solves it. | They are serious alternatives. Win one supported workflow, not a checklist comparison. | Buyers consistently prefer the alternative and cannot identify a missing operational capability worth paying for. |
| Condenser is an interchangeable agent UI. | Use it to expose task/compute evidence; generic chat panes are not the paid differentiation. | Users like the UI but neither renew service nor request operational features. |
| Services distract from the product. | Standardize a deliverable, cap concurrent pilots, and productize recurrent needs. | Delivery exceeds scoped hours or every customer requires a different backend/model. |
| Community supply is unreliable or unused. | Start with one buyer workload and a small permissioned fleet; price accepted output with retries included. | No repeat volume clears at a price covering hosts, verification, support and risk. |
| Public-benefit economics are unsustainable. | Separate software/services from a proposed marketplace split and publish costs/reserves. | The operations allocation cannot cover costs at realistic utilization, or governance cannot be documented. |

## 4. Competitive research: the buyer's alternatives

Checked **2026-10-30**. Vendor functionality is reported, not independently exercised. This is a choice set, not an exhaustive market census.

| Alternative | Observed/reported offer | Buyer reason and Kadmium implication |
| --- | --- | --- |
| Ollama local + cloud | Local use, PAYG, $20 Pro/$100 Max/$500 Team; allowances, concurrency and model-dependent prices are explicit. [S1] | Familiar local-to-cloud workflow. Hybrid access and included credits are not differentiation; do not copy allowances without understanding costs. |
| LM Studio / Bionic / LM Link | Free local agent/model features and LM Link for up to five devices; $20 Bionic+ and $100 Pro. LM Link accesses another device's model. [S2] | Polished UX and remote access. Distinguish this from sharding one model; either may satisfy the customer's job. |
| Exo / local.ai | Model clustering, business sales, hardware/model discovery, Macs/Linux. RDMA has specific Mac/Thunderbolt requirements. [S3] | The nearest strategic competitor. Mixed-device setup must win on an actual configuration; a calculator is not a moat. |
| GPUStack | Cluster management, governance, serving backends and enterprise capabilities. Version 2.1 requires Linux workers, with WSL2 guidance. [S6–S7] | Serious fleet-operations alternative. Native Windows/macOS could matter, but must be demonstrated; Linux-only is version-specific. |
| SaladCloud | USD pricing and cards alongside optional crypto; its site also advertises private deployment on customer GPUs. [S4–S5] | Existing supply and workload delivery. USD/private compute are not empty categories. Listed GPU changes take effect **November 1**, not October 30. |
| Prime Intellect | Compute, inference and RL tooling; July announcement reports $130M raised and >$100M annualized revenue. [S8] | Integrated stack and expertise. Category interest is not Kadmium demand or audited ARR; integrate where useful. |
| Tailscale + existing runtime | Free personal use and paid governance/connectivity; published $8 Standard/$18 Premium per-user prices. [S9] | A practical DIY alternative. A governance business is plausible, but connectivity alone is insufficient. |
| API batch, a larger workstation, no change | The buyer's actual workflow, equipment and supplier invoices | Less migration and known quality. Include these in every assessment; recommending the existing setup is a valid result. |

**Research changes the thesis:** pursue a specific underserved workflow inside a competitive market. Do not argue that all competitors abandoned local/private/fiat compute. Funding announcements establish neither our customer count nor achievable margin.

## 5. Demand, market sizing and pricing evidence

The 2025 Stack Overflow survey reports more respondents distrusting AI accuracy than trusting it (46% versus 33%). This supports testing reproducibility and quality, not assuming developers will pay for another harness. It is a self-selected developer survey, not a sample of Kadmium buyers. [S12]

There is no defensible TAM estimate in the available evidence. Use a **bottom-up reachable-account exercise**: identify 100 real engineering teams with an operator, recurring open-model task, hardware permission and an appropriate contact route. Record budget authority, incumbent and trigger. If 20 have a qualified discussion, ten request proposals, six buy pilots and three retain service, month-three recurring run-rate is $897 at the proposed $299 price. These are assumptions, not measured conversion or market size. Expand the list after learning why accounts bought or declined.

Interview around the last actual incident: frequency, operator hours, invoices, tolerable latency, quality and who approves spending. Ask for a paid scope and a renewal date. Praise, stars and letters of interest are weaker evidence than payment followed by continued use.

**Comparison protocol:** hold model/version, quantization, task set, input/output mix, cache state, concurrency, quality threshold, region and measurement window constant. Include retries, queueing, loading, operator work, storage, networking and power. Compare cost per accepted task. A WASI “rollout” needs CPU/memory/time limits, image, verifier and success criteria. Several shards processing one token do not create several buyer outputs.

Seeding is vulnerable to a false baseline: Cloudflare R2 has no egress-bandwidth charge, though storage/operations are charged. A $0.008/GiB seeding fee is not universally cheaper. [S13] Likewise a future GPU rate or low-priority offer is not guaranteed, available, approved wholesale capacity.

## 6. Recommendations

1. **Customer-owned engineering workflows first.** Small engineering/evaluation teams are the initial cohort; labs are secondary. Defer regulated verticals, government procurement and production-critical SLAs.
2. **Service-assisted product entry.** Test a two-week, $1,500 assessment/integration pilot with twelve delivery hours, one workload and acceptance evidence. Trial $299/month operations/support afterward. Compute accounts and spend remain with the buyer.
3. **Develop through earned requirements.** Keep native heterogeneous execution as a strategic bet. Focus on one repeatable configuration; publish correctness/capacity before speed promises.
4. **Demand before marketplace supply.** One funded batch workload on a permissioned fleet precedes anonymous residential recruitment. Require accepted-work economics, repeat orders, settlement and dispute procedures.
5. **Proof before discounts.** Reproducible reports and consented case studies beat unfunded credit giveaways. Offer bounded implementation/support.
6. **Precise mission claims.** Reuse may avoid a purchase; it does not establish lower total energy, water or emissions. IEA's 415 TWh/945 TWh figures cover all data centres and a projection, not Kadmium's avoided footprint. Measure energy per accepted task against a stated counterfactual. [S15, S18]
7. **Operate from obligations and cash.** Separate balances, earned revenue, host payables, grants and unrestricted cash. T+7 does not eliminate chargebacks. Charge acquisition time to CAC, delivery to service costs and R&D to fixed expenses without double counting. [S11]

The [business plan](BUSINESS_PLAN_AND_STANCE.md), [marketing experiments](MARKET_RESEARCH_AND_LAUNCH_PLAN.md), [operating plan](OPERATING_PLAN.md), and [economics](business/ECONOMICS.md) make these reviewable.

## 7. Primary-source register

All accessed **2026-10-30**. Live pages are dated observations, not immutable quotes or verified vendor performance. Search discovered sources; conclusions use primary pages. Third-party CAC estimates, influencer audience guesses and private-company valuation estimates do not enter the model.

| ID | Source and date/version | Evidence used / limitation |
| --- | --- | --- |
| <a id="s1"></a>S1 | [Ollama pricing](https://ollama.com/pricing), live | Prices, allowances, PAYG and concurrency; recheck before comparison. |
| <a id="s2"></a>S2 | [LM Studio pricing](https://lmstudio.ai/pricing), [LM Link docs](https://lmstudio.ai/docs/developer/core/lmlink), live | Offer and remote-model access; not arbitrary distributed sharding. |
| <a id="s3"></a>S3 | [Exo Labs](https://exolabs.net/), live | Business motion, local.ai, Macs/Linux and RDMA qualifications; vendor-reported. |
| <a id="s4"></a>S4 | [Salad pricing](https://salad.com/pricing/), live; GPU changes effective **2026-11-01** | USD priority pricing and private-cloud offer; future rates are not October invoices. |
| <a id="s5"></a>S5 | [Salad billing](https://docs.salad.com/general/explanation/billing), live | Cards remain available alongside optional USDC/USDT/RENDER payments. |
| <a id="s6"></a>S6 | [GPUStack](https://gpustack.ai/), live | Vendor orchestration/governance description; not our product test. |
| <a id="s7"></a>S7 | [GPUStack installation](https://docs.gpustack.ai/2.1/installation/installation/), **v2.1** | Worker requirements; prefer versioned instructions over older wiki statements. |
| <a id="s8"></a>S8 | [Prime Intellect Series A](https://www.primeintellect.ai/blog/series-a), **2026-07-08** | Company-reported funding/annualized revenue; not audited recurring revenue or our demand. |
| <a id="s9"></a>S9 | [Tailscale pricing](https://tailscale.com/pricing), live | Governance/connectivity anchor, not compute on its own. |
| <a id="s10"></a>S10 | [Stripe US pricing](https://stripe.com/en-us/pricing), live | 2.9% + $0.30 domestic-card benchmark; excludes other products, international and dispute costs. |
| <a id="s11"></a>S11 | [Stripe disputes](https://docs.stripe.com/disputes/how-disputes-work), live | Balance debits; typical initiation within 120 days, sometimes longer. |
| <a id="s12"></a>S12 | [Stack Overflow AI survey](https://survey.stackoverflow.co/2025/ai), **2025** | Trust/quality signal; self-selected, not willingness to pay or a 2026 census. |
| <a id="s13"></a>S13 | [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/), live | No egress-bandwidth fee; storage/operations/retrieval still matter. |
| <a id="s14"></a>S14 | [HHS HIPAA cloud guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html) | Encryption/no access to keys does not itself remove business-associate obligations; analyze the deployment. |
| <a id="s15"></a>S15 | [FTC Green Guides](https://www.ftc.gov/business-guidance/resources/environmental-claims-summary-green-guides) | Specific, qualified substantiation for environmental claims. |
| <a id="s16"></a>S16 | [FinCEN prepaid-access FAQ](https://www.fincen.gov/resources/statutes-regulations/guidance/final-rule-definitions-and-other-regulations-relating), **2011-11-02** | Bounded closed-loop exclusion, not blanket approval for credits, bonds, expiry or payouts; other/state rules remain separate. |
| <a id="s17"></a>S17 | [SEC Regulation Crowdfunding](https://www.sec.gov/resources-small-businesses/exempt-offerings/regulation-crowdfunding) | Intermediary/offering requirements; not operating cash assumed in this plan. |
| <a id="s18"></a>S18 | [IEA Energy and AI](https://www.iea.org/reports/energy-and-ai/executive-summary), **2025 report** | Data-centre estimate/projection; no proof of Kadmium savings. |
| <a id="s19"></a>S19 | [Show HN guidelines](https://news.ycombinator.com/showhn.html), live | A usable project, not landing page/fundraiser; no solicited votes. |
| <a id="s20"></a>S20 | [FTC endorsements](https://www.ftc.gov/business-guidance/advertising-marketing/endorsements-influencers-reviews), live | Disclose material creator/affiliate relationships. |
| <a id="s21"></a>S21 | [NSF 26-510](https://www.nsf.gov/funding/opportunities/small-business-innovation-research-small-business-technology/nsf26-510/solicitation), **2026** | Phase I Pitch invitation, PI employment and STTR partner requirements. Recheck eligibility/deadline; assume no award. |

## 8. Research gaps and next evidence

- **Demand:** founder owns 20 interviews and an account ledger; record no-purchase reasons and the incumbent, including no change.
- **Product:** engineering owns the supported setup, matched results, install/rollback evidence and provider billing boundaries.
- **Economics:** finance owns actual invoices, delivery hours, refunds, renewals, processor terms and cash; replace model inputs after each pilot.
- **Rights/governance:** founder owns entity/charter decisions, license files, dependency/model redistribution review and brand/domain review. No golden-share, audited-compliance or charitable-receipt claim without documentation.
- **Competition:** refresh S1–S9 monthly and before a campaign; private offers may require a quote and evaluation.
- **Publication:** /site, README capability claims, companion/dashboard quotes and runtime economic examples need a separate implementation pass. The P0 publication blocker remains open; approving this strategy does not make those surfaces ready for promotion.
