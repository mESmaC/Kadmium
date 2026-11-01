# Kadmium plan stress test — business plan and marketing plan (31 October 2026)

**Scope.** This review stress-tests version 3 of the [business plan](BUSINESS_PLAN_AND_STANCE.md) and the [marketing and launch plan](MARKET_RESEARCH_AND_LAUNCH_PLAN.md), together with the [operating plan](OPERATING_PLAN.md) and the [economics model](business/ECONOMICS.md), as they stand on `main` at `420cad5`. It judges them against the **direction of the project**: a mixed-vendor local compute fabric (Windows, macOS, Linux, ARM) with Condenser as the daily workspace, overflow to Kadmium Cloud, and a USD-metered Supply Grid, reached through the version 3 bridge of a paid pilot and a monthly service. It assumes the public release, licence files, tagged artifacts and web and email surfaces that the plans already require exist as planned; it does not re-audit the repository's present state, which the [evidence review](MARKETING_BUSINESS_REVIEW_2026-10-30.md) and the [strategy audit](STRATEGY_AUDIT_2026-10-30.md) cover.

**Method.** Desk research checked on **2026-10-31**, repository inspection, and sensitivity runs through the checked-in [`business_model.py`](../scripts/business_model.py) with alternative inputs (Appendix A). No customer interviews, revenue records or legal advice. Labels follow the evidence review: **observed** for repository content or a source's own statement, **reported** for third-party or vendor figures, **derived** for arithmetic, **hypothesis** for proposals. Vendor and consultancy sources are marked as such; several conflict, and the conflicts are shown rather than averaged.

**Short verdict.** Version 3 fixed the honesty problem and most of the scope problem. It has not fixed the economics problem, and it has introduced a sequencing problem. The service bridge cannot reach break-even at the capacity the operating plan allows, at any price the plan proposes, and at market labour rates every pilot loses money. The bridge also leads away from the destination: the pilot as scoped uses none of the project's differentiator, competes with free tools and with coding agents, and produces no evidence toward the native multi-device proof (G3) that defines the category. The market evidence gathered here says the category gap is real and still open, that buyers' stated pain is the self-hosting skill gap, and that the money in open-weight deployment is captured by infrastructure and platform sellers rather than by service hours. The recommended change is an inversion of the gate order, not a return to the superseded plan: measure the mixed-vendor pair first, sell design-partner discovery only to buyers with a multi-machine constraint, and keep software and managed tiers as the revenue model to test once the measurement exists.

---

## 1. Verdict in one page

| # | Finding | Severity | Status in version 3 | Section |
| :--- | :--- | :---: | :--- | :--- |
| 1 | The service bridge cannot break even at the modelled capacity: at full utilisation of the plan's own hours the planning assumptions yield about $4,300 of monthly contribution against $5,750 of fixed cost. | P0 | Not addressed; the plan says it "loses money" but not that it cannot stop losing money. | §2 |
| 2 | At market labour rates for the work described ($150–300/h reported), the $1,500 pilot's contribution is negative; at $75/h a modest scope overrun makes it negative. | P0 | Labour is modelled at $75/h with no sensitivity. | §2.2 |
| 3 | The pilot does not exercise the differentiator and competes with free interfaces, free evaluation harnesses, vendor-native local runtimes and coding agents. | P1 | Acknowledged as an objection; not resolved by the scope. | §3 |
| 4 | The gate order is inverted for a product company: six pilots of single-node setup produce no G3 evidence, while the mixed-vendor gap is still open and vendor platforms are closing adjacent gaps. | P1 | G3 is gated behind G1/G2. | §3.3, §4 |
| 5 | Demand evidence supports a product that removes self-hosting skill cost, not hourly services; value accrues to infrastructure and platform sellers. | P1 | Plan treats services as the revenue line. | §5 |
| 6 | The marketing plan's positioning line does not name the project's direction, so the audience the content attracts (people pooling machines) and the offer sold (single-node repeatability) do not match. | P1 | Deliberate in version 3; the cost is unaddressed. | §6.1 |
| 7 | M1's arithmetic (8 discussions from 40 contacts in 12 hours) requires a warm list the plan does not identify; cold benchmarks are 1.4–3.4% reply. | P2 | Thresholds set; source of the 40 is not. | §6.2 |
| 8 | M6's $300 ad test buys roughly 30 clicks and at most one lead at reported LinkedIn costs; it cannot produce a decision. | P2 | Budget cap set; learning value is nil. | §6.5 |
| 9 | Costs missing from the model: insurance, legal, accounting, review hours inside the $299 service, and collection delay. | P2 | Excluded with a note. | §2.5 |
| 10 | The plan's strongest assets (labelled evidence, buyer-paid compute, capacity caps, stop rules, no marketplace before demand) hold under stress. | Keep | — | §7 |

---

## 2. The service bridge cannot carry the company

### 2.1 Capacity ceiling (derived)

The operating plan reserves 16 commercial hours a week and caps active pilots at two, each twelve delivery hours over two weeks. That fixes the ceiling. At 16 hours × 52 weeks ÷ 12, there are **69.3 commercial hours a month**. Two concurrent two-week pilots allow at most **four pilots a month**. Using the checked-in planning assumptions ($75/h labour, 12 delivery hours, one support hour per account-month) and a generous ten acquisition hours a month (the planning scenario uses about fifteen):

| Line | Hours | Units | Contribution |
| :--- | ---: | ---: | ---: |
| Pilots at $1,500 (12 h each) | 48.0 | 4 | $2,104.80 |
| Acquisition | 10.0 | — | $0.00 |
| Recurring accounts at $299 (1 h each) | 11.3 | 11.3 | $2,199.23 |
| **Total contribution** | **69.3** | | **$4,304.03** |
| Fixed product/admin and overhead (model) | | | $5,750.00 |
| **Monthly result at full utilisation** | | | **−$1,445.97** |

Revenue at that ceiling is about $9,400 a month. The ceiling cannot be raised by selling more: every additional pilot displaces support hours and every additional account displaces pilot hours. With sixteen-hour pilots, four pilots consume 64 of the 69.3 hours and there is no support capacity at all. The model's planning scenario (121 of 208 hours over thirteen weeks) is therefore not a ramp toward profitability; it is a point on a curve whose maximum is below the fixed cost line.

Three things could change this and none of them is in the plan: a higher price per delivered hour, a product that earns without hours, or a fixed cost funded by something other than service contribution. The plan's own note that the planning case "tests whether to invest, not whether profitability can be announced" is correct, but a plan that cannot reach break-even at full utilisation is not testing investment in services; it is testing whether paid discovery is worth running. It should say so.

### 2.2 Labour rate (reported, derived)

The model values founder delivery time at $75/h. Published 2026 rate guides for the work the pilot describes (configuring inference backends, evaluation harnesses and runbooks for a team) cluster at **$150–300/h** for an experienced practitioner, with marketplace medians near $100/h and senior freelancers at $150–240/h; the sources are consultancies and marketplaces with a stake in the figures, and they conflict [S9][S10]. Re-running the planning scenario at $150/h:

| Scenario (planning cohort unless noted) | Contribution per pilot | Contribution per account-month | 90-day operating result |
| :--- | ---: | ---: | ---: |
| As checked in ($75/h, 12 h, 1 support h) | $526.20 | $194.05 | −$17,891.60 |
| $150/h | **−$373.80** | $119.05 | −$26,966.60 |
| $150/h, 16 pilot h, 2 support h | **−$973.80** | **−$30.95** | −$31,166.60 |
| $75/h, 20 pilot h, 2 support h | **−$73.80** | $119.05 | −$21,791.60 |
| $75/h, pilot price $3,000 | $1,952.70 | $194.05 | −$9,332.60 |
| $150/h, pilot price $3,000, 16 h | $452.70 | $119.05 | −$22,007.60 |
| $150/h, pilot $4,500, 16 h, service $499 with 2 h | $1,879.20 | $159.25 | −$13,287.80 |

At the market rate the pilot is a loss on every unit before acquisition. At the plan's own rate, a scope overrun to twenty hours, which is ordinary for a first engagement that promises a baseline, a provider configuration, a reproducibility report, a runbook and a go/no-go, also turns it negative. The plan's benchmark acceptance standard (at least five complete runs per configuration, matched baselines, failure counts) cannot be met inside twelve hours; either the pilot's report is weaker than the standard or the hours are higher than the model.

### 2.3 Price anchors (reported)

Small-business AI pilots and discovery engagements are quoted by vendors in 2026 at roughly **$3,000–15,000** for assessment and scoping and **$5,000–15,000** for a pilot build, with larger firms at $15,000–40,000 for discovery; all are vendor pages and the spread is wide [S11][S12]. At $1,500 for twelve hours ($125/h), the pilot is priced at the bottom of the market. That is a legitimate choice for paid discovery. It is not a price that funds a product, and it selects for buyers whose budget authority tops out near $1,500, which the plan's own qualification ("who can approve a $1,500 assessment") then confirms.

### 2.4 The recurring unit (derived)

The $299 service promises "a monthly cost and reliability review" and "up to one support hour". The model costs one hour. If the review takes an hour, contribution per account-month falls from $194 to $119; at two support hours plus a review it is $44. The service's economics depend on the review being free, which the operating plan's weekly-rhythm table does not say.

### 2.5 Costs the model leaves out (reported, derived)

- **Insurance.** Clients who let a vendor configure machines and credentials commonly require technology errors-and-omissions and cyber cover. Insureon's medians for IT consultants are **$899/yr** for tech E&O, **$1,540/yr** for cyber and **$369/yr** for general liability [S13]. About $230 a month, or 1.2 recurring accounts' contribution.
- **Legal and accounting.** A statement-of-work template, data-handling terms, and the accountant the plan already names for recognition policy are cash costs before the first pilot.
- **Collection delay.** The model collects every invoice in the month delivered. B2B buyers of a $1,500 assessment often pay on net-30 terms; the 13-week cash forecast the operating plan requires should carry that lag as its base case, not a stress.
- **Review hours.** See 2.4.

None of these changes the direction of the finding; they widen the gap.

---

## 3. The bridge does not lead to the destination

### 3.1 What the pilot actually sells, and its free substitutes (observed, reported)

The pilot's supported configuration is "already-operational local inference plus Condenser": Ollama, a provider configuration, a compute-inspection snapshot that previews placement without executing, lossless checkpoint normalisation, and a runbook. Each piece has a free, mature substitute that the buyer's team can already find:

| Pilot component | Free substitute the buyer already has or can install in minutes | Status (reported, checked 2026-10-31) |
| :--- | :--- | :--- |
| Condenser as the task workspace | Open WebUI, LibreChat, AnythingLLM, Jan | Free, large communities; star counts conflict across sources but are tens of thousands and up [S14] |
| "Recurring open-model evaluation" harness | promptfoo, Inspect AI, lm-evaluation-harness | MIT-licensed, actively maintained, standard in 2026 evaluation practice [S15] |
| Local inference backend on the buyer's machines | Ollama, LM Studio (free for work since July 2025, LM Link for remote access), Microsoft Foundry Local (GA April 2026, Windows ML integration) | Vendor-native, free [S3][S6][S7] |
| Runbook and provider configuration | The team's own coding agent | 66% of Stack Overflow 2026 AI users use coding agents, 73% daily [S8] |

The last row is the one the plan does not address. Twelve hours of "configure, baseline, document" is the shape of task that coding agents now do for a team in an afternoon. The value that survives is the **trusted decision**, which the plan names correctly, and the **mixed-device placement** that no free tool gives them, which the pilot explicitly excludes until G3.

### 3.2 The pilot generates no evidence toward G3

The business plan says native execution is "an extension only after verification" and that the pilot "does not promise new GPU kernels in twelve hours". Both are right. The consequence is that six pilots of single-node setup, however well delivered, teach nothing about the only capability that distinguishes Kadmium from the table above. The plan's path to G3 is "a repeated buyer memory/placement constraint and verified backend". The first half can only be observed if pilots are selected for buyers who already have that constraint; the qualification criteria in business plan §3 do not select for it, and the disqualifiers (unsupported hardware) select against it.

### 3.3 The gate order is inverted for this company

Version 3 orders the gates G0 (credible evaluation) → G1 (three paid pilots) → G2 (retention) → G3 (native proof). For a services company that order is right: prove people pay, then invest. For a product company whose category is defined by one measurement, it delays the one asset that creates demand for everything else, and it spends the founder's scarce commercial hours (§2.1) on work that cannot become a product. The strategy audit's Gate 0 ("a reproducible two-node measurement against llama.cpp RPC exists") is the correct first gate for the direction; version 3's status table noted it as "the G3 form" and deferred it. This review's recommendation (§8) is to restore it.

---

## 4. The direction's gap is real, still open, and closing at the edges

Checked **2026-10-31**. Vendor functionality is reported from the vendors' own pages, not exercised.

| Alternative | What it serves today | What it leaves open for Kadmium | Movement to watch |
| :--- | :--- | :--- | :--- |
| Exo | **Tier 1: Apple Silicon macOS only.** Linux CUDA and Linux CPU are "planned" Tier 1; Linux Vulkan/CUDA on Framework Desktop are planned Tier 2; Windows CUDA and CPU are "Longer term" [S1]. | Windows and NVIDIA consumer GPUs; mixed-vendor pairs; polite yield; overflow. | A community fork adds CUDA via tinygrad [S2]. When Exo's planned Linux CUDA ships, the open gap narrows to Windows and mixed pairs. |
| Apple MLX with JACCL | RDMA over Thunderbolt 5 in macOS 26.2, demonstrated on four M3 Ultra Mac Studios at WWDC26 [S4]; hands-on reports of about 14 tok/s on a 70B dense model across four Mac Studios (secondary, single tester) [S5]. | Everything that is not an all-Apple cluster. | Mac-only pooling is now vendor-native. Kadmium's Mac story must be "the Mac plus the PC", never "the Macs". |
| GPUStack | "Only Linux is supported for GPUStack worker nodes"; "macOS is not supported for GPUStack worker nodes"; Windows users are pointed to WSL2 [S16]. | Native Windows and macOS workers; consumer hardware. | Datacenter-shaped; unlikely to move toward gaming PCs. |
| Ollama | v0.40.13 on 28 October 2026; no multi-node or cluster inference in recent release notes [S3]. | Multi-machine execution. | Ollama is the pilot's own backend; if Ollama adds LAN pooling the pilot's configuration becomes the competitor's. |
| Microsoft Foundry Local | GA April 2026; Windows ML execution-provider acquisition through the OS [S6]. | Multi-machine; non-Windows. | Single-machine Windows inference is becoming an OS feature. The free tier cannot differentiate on single-box Windows. |
| llama.cpp RPC | Mainline; AMD publishes a four-node guide; community two-node runs on identical GPUs over 2.5GbE [S17]. | Scheduling, yield, verification, overflow, mixed-OS results. | This is the free incumbent for pooling and the comparison the measurement must beat or match honestly. |
| Rust heterogeneous projects (cake, swarmllm, Ghostlink) | Experimental; cake shards across iOS, Android, macOS, Linux and Windows and describes itself as experimental and fast-changing [S18]. | Maturity, verification, economics. | Proof that the idea is reachable in Rust by a small team; also proof that "Rust distributed inference" is not by itself a moat. |

Reading: the specific configuration the project's own benchmark ledger used, a Windows RTX PC and an M4 Mac mini, is the configuration no vendor serves and the one the audience most often owns. The window is open because Apple serves Apple, Microsoft serves one Windows box, Exo serves Macs with Linux next, and GPUStack serves Linux racks. It is closing at the edges: each vendor is absorbing the single-vendor case natively, so the only durable claim is the mixed-vendor one, with ownership, yield and overflow around it. Version 3 postpones the only work that makes that claim.

---

## 5. Demand evidence supports a product, not a service

Three sources, each with limits, point the same way.

- **Open-weight adoption is a minority and moving.** Menlo Ventures' mid-2025 survey reported that 13% of enterprise daily workloads used open-source models, down from 19%, and that more than half of enterprises used none [S19]. Futurum's September 2026 drill-down of 200 enterprises reported that 42% run open-weight models in production, up from 31% in July, with hosting led by self-managed cloud (46%) and on-premises (42%) [S20]. The two surveys measure different things (share of workloads versus share of enterprises) in different years; neither is a sample of three-to-fifteen-person engineering teams that own hardware. Together they say the segment exists and is not the majority.
- **The pain is skill and cost of self-hosting, not lack of interest.** Futurum: 52% of non-users say they lack self-hosting and fine-tuning skills; self-hosting costs now stall 50% of pilots; in-house open-weight builds report sustained ROI at scale in only 10% of cases [S20]. That is the pain the version 3 pilot addresses, and it is the argument for a product that removes the skill requirement rather than a service that rents the skill by the hour.
- **Value accrues to infrastructure and platform sellers.** Futurum's framing is that token volume moving to open weights "leaves the AI budget and reappears as infrastructure spending" [S20]. Stack Overflow 2026 reports that tool switching is driven by output quality (25.1%) far more than cost (6.4%) [S8]. A product that makes owned hardware serve a model at full quality is positioned where the spend goes; an hourly assessment is not.

The business plan's statement that "the attractive customer has a recurring open-model evaluation/development task, owns useful machines, and spends operator time on setup" is consistent with this evidence. Its conclusion that the way to serve that customer first is twelve billed hours is not.

---

## 6. Marketing plan under stress

### 6.1 Positioning does not name the direction

"Understand your compute. Make your AI workflow repeatable." is a line any observability or MLOps vendor could use. It does not say what Kadmium is for, which machines it is for, or what the buyer searches for. Version 3 chose it deliberately to avoid promising unmeasured pooling, and the operating plan's claim register is the right discipline. The cost is a message-to-offer mismatch: the content programme (configuration reports, "bring your awkward fleet", the two-machine write-up, hardware-tribe venues) attracts people who want to pool machines, and the offer sold to them is single-node repeatability. The honest fix is not to hide the direction but to state it with its status in one sentence, as the voice guide already models ("Target: … Current provider integration is planned."), and to sell the pilot only to the subset whose constraint is multi-machine.

### 6.2 M1 arithmetic (reported, derived)

M1 asks for eight discussions from forty "tightly qualified" teams in two weeks and twelve hours. Reported cold-email reply benchmarks for 2026 are **1.4–3.4% average** depending on source, with 5–10% described as good and 10%+ as excellent on tight segments; the sources are outreach-tool vendors and conflict [S21][S22]. A 20% discussion rate from forty contacts is a warm-network rate. The plan does not say where the forty come from. If they are cold, forty contacts yield one reply and the experiment's "fewer than four discussions" stop rule fires on the first attempt. If they are warm, the real constraint is the founder's network size, and the plan should name the list before counting the outcome. Twelve hours also has to cover researching and personalising forty messages and holding eight twenty-minute discussions with preparation; it is tight.

### 6.3 M2 needs an audience

Two clinic sessions with "six relevant attendees" presuppose a community, a mailing list or a partner venue. None is named. Before public release the project has no community to invite; after it, the clinic is a good format and should be sequenced after the first community post, not alongside M1.

### 6.4 M4's window is too short

Two comparison pages are expected to produce impressions and ten engaged visits in six weeks. Organic search for new pages on a new domain typically takes months to index and rank; six weeks measures indexing, not demand. Set the window to a quarter or treat the pages as sales collateral first.

### 6.5 M5 and M6 cannot produce decisions (reported, derived)

- M6's $300 ad test, at Metadata's 2025 LinkedIn medians of **$9.39 per click** and **$202 per lead** across 138 B2B advertisers [S23], buys about thirty clicks and at most one lead. One lead cannot be attributed, retained or paid back. Either fund a test that can produce a denominator (a few thousand dollars) or cut M6 and keep the budget.
- M5's $400 for an independent creator reproduction will not buy a creator with the target audience's reach; the plan's own instruction to "obtain a quote first" is right, and the budget line should be left blank until then rather than anchored at $400.

### 6.6 The trust surface is thin for a paid engagement

A $1,500 engagement that touches a team's machines and credentials is bought from a person. The draft page copy has no founder name, no company identity, no calendar link and a single email call to action. The evidence review's consent and attribution rules (C7) cover what not to publish; the plan should also say what to publish: who is delivering, their background, and a direct way to book a conversation.

### 6.7 The ninety-day calendar is overcommitted

Days 1–14 contain publication remediation, choosing the supported workflow, preparing the SOW and baseline, running M1 and a clinic, alongside 24 engineering hours a week. The capacity arithmetic in §2.1 applies here too; the first fortnight as written needs more than the hours reserved for it.

### 6.8 What is right in the marketing plan

The north-star metric (weekly active customer teams completing their agreed workload at the accepted quality threshold), the two activation events, the no-telemetry default, the Show HN and FTC rules, the referral cap, the "CAC is not yet estimable" rule and the insistence on denominators are all sound and should be kept.

---

## 7. What holds under stress

Version 3's discipline is its asset, and this review does not weaken it:

- Every figure labelled; the 3.98× headline withdrawn; matched baselines required before any speed claim.
- Compute billed through the buyer's own provider account, so no credit liability, no prepaid float and no marketplace reserve before demand exists.
- Capacity caps, a waiting list instead of overtime, and stop rules at days 30, 60 and 90.
- Managed compute, host payouts, hardware, regulated verticals and grants deferred to gates with named evidence.
- The 75/20/5 split kept as a design objective, not a charter.
- The alternatives guide that recommends "keep your current setup" as a valid result.

The stress test's criticism is of the bridge's economics and sequence, not of the honesty that produced it.

---

## 8. Recommended changes, ranked

1. **Restore the measurement as the first gate.** Make a committed-harness, matched-baseline measurement of one mixed-vendor pair (a Windows RTX PC and an M4 Mac mini, against llama.cpp RPC on the same two machines, on a named model, with power and failure logs) the day-1 engineering objective and the first public artifact. This is the strategy audit's Gate 0 and the operating plan's benchmark acceptance standard applied to the project's own rig. Nothing else in either plan creates a category.
2. **Rename the pilot as paid design-partner discovery and cap it at three.** Qualify only buyers whose constraint is multi-machine (a model that does not fit one box they own, or a rate-limit or memory incident they can show). Price it at no less than $3,000 or reduce it to a four-hour compute-fit review at a price that is positive at $150/h; do not run twelve-hour engagements at $1,500. State in the plan that this line funds learning, not the company.
3. **Remove the $299 service from the plan until G1 produces a repeat job.** Version 3 already restricts continuing work to pilot extensions before G1; the model should not carry a recurring line that the plan does not yet sell.
4. **Adopt the software-tier spine as the revenue hypothesis to test at G3**, not before. The strategy audit's software-only base case ($199 team tier, $499 office tier) breaks even against the model's $5,750 fixed cost at roughly thirty team subscriptions or twelve office subscriptions (derived), which is a reachable number for a measured, open-core tool and an unreachable one for an hourly service. Test the price only against buyers who have seen the measurement.
5. **Fix the economics model.** Add sensitivity rows for labour at $75 and $150, pilot hours at 12, 16 and 20, review hours inside the service, insurance, legal and accounting lines, net-30 collection, and a capacity-ceiling row that reports the best month the plan allows. Keep the no-sales case.
6. **State the direction in the positioning, with its status.** One sentence that names mixed-vendor pooling as the target and the present boundary, then the offer. Sell the pilot only to the audience that sentence attracts.
7. **Repair M1, M2, M4, M5 and M6.** Name the warm list before counting discussions; sequence the clinic after the first community post; give M4 a quarter; leave M5's budget blank until quoted; cut M6 or fund a test with a denominator.
8. **Put a person on the page.** Founder name, background, a booking link and the delivery boundary.
9. **Decide the platform question in writing.** Apple now pools Macs natively and Microsoft serves one Windows box natively. Kadmium's claim must be the mixed pair with ownership, yield and overflow around it. Everything single-vendor is a feature to interoperate with, not a market to win.

Items 1, 2, 5 and 6 are changes to the plan documents; items 3, 4, 7, 8 and 9 are decisions the founder owns. None of them touches runtime constants, model-quality defaults or any claim covered by AGENTS.md §2.1.

---

## 9. What would prove this review wrong

| Claim in this review | Evidence that would falsify it |
| :--- | :--- |
| The service bridge cannot reach break-even at the allowed capacity. | Three pilots delivered inside twelve hours with reports that meet the operating plan's benchmark standard, at a price buyers pay without negotiation, with the model's fixed cost covered by something other than contribution. |
| Pilots produce no G3 evidence. | Two of the first three pilots surface a multi-machine constraint the buyer will fund a design-partner milestone to solve. |
| The pilot competes with free tools and coding agents. | Qualified buyers who have tried Open WebUI, a free harness and their own agent still pay for the assessment and name a reason that is not "we lacked time". |
| The mixed-vendor gap is still open. | Exo, Ollama or llama.cpp ship a supported Windows-plus-Mac pooling path with a published matched benchmark before Kadmium's measurement exists. |
| M1's rate needs a warm list. | Forty cold contacts produce eight discussions. |
| The positioning mismatch costs conversion. | Visitors arriving from pooling content convert to single-node pilots at the plan's thresholds. |

---

## Appendix A — Reproducing the sensitivity figures

All figures in §2 derive from the checked-in model with changed inputs; no repository file is modified. From the repository root, with Python 3.12:

```python
import copy, json, sys
sys.path.insert(0, "scripts")
import business_model as bm
from decimal import Decimal as D

base = json.load(open("docs/business/assumptions.json", encoding="utf-8"))

def run(label, svc=None, scen=None):
    inp = copy.deepcopy(base)
    if svc:
        inp["service"].update(svc)
    s = copy.deepcopy(next(x for x in inp["scenarios"] if x["name"] == "Planning"))
    if scen:
        s.update(scen)
    r = bm.scenario_result(inp, s)
    print(label, bm.usd(r["pilot_unit"]), bm.usd(r["recurring_unit"]), bm.usd(r["result"]))

run("checked in")
run("labor 150", svc={"labor_hourly": "150"})
run("labor 150, 16h, 2h", svc={"labor_hourly": "150"},
    scen={"pilot_hours": "16", "support_hours_per_account_month": "2"})
run("labor 75, 20h, 2h", scen={"pilot_hours": "20", "support_hours_per_account_month": "2"})
run("labor 75, pilot 3000", svc={"pilot_price": "3000"})
run("labor 150, pilot 3000, 16h", svc={"labor_hourly": "150", "pilot_price": "3000"},
    scen={"pilot_hours": "16"})
run("labor 150, pilot 4500, 16h, service 499, 2h",
    svc={"labor_hourly": "150", "pilot_price": "4500", "recurring_price": "499"},
    scen={"pilot_hours": "16", "support_hours_per_account_month": "2"})

# Capacity ceiling: 16 commercial hours a week, two concurrent two-week pilots.
hours = D(16) * D(52) / D(12)              # 69.33 h/month
s = next(x for x in base["scenarios"] if x["name"] == "Planning")
r = bm.scenario_result(base, s)
pilots, acquisition = 4, D(10)
support = hours - pilots * D(s["pilot_hours"]) - acquisition
accounts = support / D(s["support_hours_per_account_month"])
contribution = pilots * r["pilot_unit"] + accounts * r["recurring_unit"]
fixed = D(base["service"]["fixed_product_admin_monthly"]) + D(base["service"]["fixed_overhead_monthly"])
print("ceiling", bm.usd(contribution), "fixed", bm.usd(fixed), "net", bm.usd(contribution - fixed))
```

Run with `python -I` from the repository root. `python scripts/business_model.py --check` passed at `420cad5` before these runs.

## Appendix B — Source register

All accessed **2026-10-31**. Live pages are dated observations. (P) primary or vendor's own page; (3P) third party.

| ID | Source | Used for / limitation |
| :--- | :--- | :--- |
| S1 | (P) Exo, `PLATFORMS.md` on `main`. https://github.com/exo-explore/exo/blob/main/PLATFORMS.md | Tier 1 Apple Silicon only; Linux planned; Windows "Longer term". |
| S2 | (3P) `Scottcjn/exo-cuda`. https://github.com/Scottcjn/exo-cuda | Community CUDA fork; not verified. |
| S3 | (P) Ollama releases. https://github.com/ollama/ollama/releases | v0.40.13, 28 Oct 2026; no multi-node notes. |
| S4 | (P) Apple WWDC26 session 233. https://developer.apple.com/videos/play/wwdc2026/233/ | RDMA over Thunderbolt 5 in macOS 26.2 with JACCL and MLX. |
| S5 | (3P) Implicator.ai on a four-Mac Studio RDMA test. https://www.implicator.ai/apple-just-turned-a-software-update-into-a-730-000-discount-on-ai-infrastructure/ | Secondary hands-on figures; single tester. |
| S6 | (P) Microsoft DevBlogs, Foundry Local GA, 9 Apr 2026. https://devblogs.microsoft.com/foundry/foundry-local-ga/ | GA and Windows ML integration. |
| S7 | (P) LM Studio, free for work, 8 Jul 2025. https://www.lmstudio.ai/blog/free-for-work ; (3P) LM Link coverage. https://www.digitalapplied.com/blog/lm-studio-locally-lm-link-iphone-local-llm-2026 | Free commercial use; LM Link availability; enterprise pricing not public. |
| S8 | (P) Stack Overflow 2026 Developer Survey, AI section. https://survey.stackoverflow.co/2026/ai | Agent usage 66%, daily 73%; switch reasons quality 25.1% vs cost 6.4%. No local-model breakout. |
| S9 | (3P, marketplace) goLance ML engineer rate guide 2026. https://golance.com/hiring/best-freelance-machine-learning-engineers-hourly-rate | Mid $118/h, senior $195/h; vendor figures. |
| S10 | (3P, vendor) Orient Software AI consultant rates. https://www.orientsoftware.com/blog/ai-consultant-hourly-rate/ | $100–300/h bands; vendor figures; conflicts with S9. |
| S11 | (3P, vendor) AI pilot project cost guide. https://sagnikbhattacharya.com/blog/ai-pilot-project-cost | $5,000–15,000 small-business pilots; vendor. |
| S12 | (3P, vendor) AI strategy cost 2026. https://mahluminnovations.com/blog/ai-strategy-cost-2026 | $15,000–40,000 assessments for larger firms; vendor. |
| S13 | (P, insurer) Insureon IT consultant insurance cost. https://www.insureon.com/technology-business-insurance/it-consultants/cost | Median tech E&O $899/yr, cyber $1,540/yr, GL $369/yr. |
| S14 | (3P) Self-hosted AI interface comparison 2026. https://dev.to/jovan_chan_9500711396d4e6/anythingllm-vs-open-webui-vs-librechat-in-2026-which-self-hosted-ai-interface-should-you-use-24cl | Free substitutes for the workspace; star counts conflict. |
| S15 | (3P) Open-source eval frameworks 2026. https://futureagi.com/blog/best-open-source-eval-frameworks-2026 | promptfoo, Inspect AI, lm-evaluation-harness status and licences. |
| S16 | (P) GPUStack README on `main`. https://github.com/gpustack/gpustack/blob/main/README.md | Linux-only workers; macOS unsupported; WSL2 for Windows. |
| S17 | (P) AMD Ryzen AI Max+ cluster guide, Feb 2026. https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html | llama.cpp RPC four-node guide; single vendor. |
| S18 | (P) `evilsocket/cake`. https://github.com/evilsocket/cake | Rust heterogeneous sharding; self-described experimental. |
| S19 | (3P) TechCrunch on Menlo Ventures mid-2025 report, 31 Jul 2025. https://techcrunch.com/2025/07/31/enterprises-prefer-anthropics-ai-models-over-anyone-elses-including-openais/ | 13% of enterprise workloads open source, down from 19%; more than half use none. |
| S20 | (3P, analyst) Futurum, 26 Oct 2026. https://futurumgroup.com/insights/enterprises-embrace-open-weights-but-infrastructure-sellers-profit-most/ | 42% of 200 enterprises run open weights in production; hosting split; 52% skill gap; 50% of pilots stalled by self-hosting cost; spend accrues to infrastructure. |
| S21 | (3P, vendor) lemlist cold email response rates, updated 28 Sep 2026. https://lemlist.com/blog/cold-email-response-rate | 3.43% platform-wide average (Woodpecker, 20M emails); 5–10% good. |
| S22 | (3P, vendor) Instantly cold email reply benchmarks, updated 1 Jul 2026. https://instantly.ai/blog/cold-email-reply-rate-benchmarks | 5–9% all-in averages cited; conflicts with S21 and with lower 1.4% figures elsewhere. |
| S23 | (3P, vendor) Metadata 2026 B2B paid media benchmarks. https://metadata.io/resources/blog/what-linkedin-ads-cost | LinkedIn CPL $202 median (138 advertisers), CPC $9.39; 153 advertisers, $57.6M 2025 spend. |
| S24 | (P) Kadmium repository at `420cad5`: business plan, launch plan, operating plan, `docs/business/assumptions.json`, `scripts/business_model.py`, `docs/BENCHMARKS_2026-10.md`. | Plan text, model inputs and the configuration used for Run 1. |
