# Kadmium business plan and stance

**Version 3 — proposed strategy, 2026-10-30.** Supersedes the launch-ready/Day-90 revenue plan. This is a proposal for review, not a price announcement, adopted corporate charter, forecast, or claim that managed services are live.

See the [evidence review](MARKETING_BUSINESS_REVIEW_2026-10-30.md), [launch experiments](MARKET_RESEARCH_AND_LAUNCH_PLAN.md), [operating plan](OPERATING_PLAN.md), and [financial model](business/ECONOMICS.md).

## 1. The business we are trying to build

**Help small AI engineering teams get reliable, understandable work from the computers they already own.** Keep local experimentation open and self-hostable; earn revenue from deployment expertise, operational convenience, and eventually tested team controls. Preserve native heterogeneous execution as the long-term product investment.

The attractive customer has a recurring open-model evaluation/development task, owns useful machines, and spends operator time on setup, placement, troubleshooting and cloud/local decisions. The buyer pays to reduce that work or enable a workload under a real constraint. Cheap APIs, existing local runtimes and a larger single machine are legitimate alternatives.

The mission remains: owners control equipment and data paths; useful reuse is preferable to unnecessary purchases; contributors understand their compensation and costs. Prove benefits at workload level. “No additional datacenters,” guaranteed income, automatic compliance and zero-cost compute are not sales promises.

## 2. What can support a sale now

| Layer | Evidence at the reviewed commit | Commercial treatment |
| --- | --- | --- |
| Condenser with a configured provider | Provider adapters, task controls, compute inspection and browser-local workflow. See [guide](guides/condenser-workspace.md). | Evaluate a supported workflow with customer-owned provider accounts; name the provider/data destination. |
| Kadmium control plane and reference execution | Worker transport/planning/reference paths exist; /v1 endpoints remain diagnostics. | Demonstrate planning honestly, not native generation or drop-in API compatibility. |
| Native multi-device execution | A reported external scratch-harness benchmark with unresolved reproducibility/configuration questions. | Strategic R&D with acceptance gates, not general availability or a speed guarantee. |
| CloudShield, spot procurement, clearinghouse | Simulations/planners/fixtures are not operational procurement, banking or SLAs. | Deferred; no prepaid compute, automatic failover or host payout launch on this evidence. |
| Office governance, regulated deployments, hardware | Packages exceed demonstrated operating/legal readiness. | Deferred; no HIPAA/CMMC certification, BAA, zero-egress, manufactured-device or grant promises. |

The [October numbers ledger](NUMBERS_2026-10.md) remains an engineering fixture reference, not a wholesale feed, current price book or authorized treasury policy. This strategy changes no runtime constants.

## 3. First customer and buying motion

**Primary segment:** AI engineering/evaluation teams, roughly 3–15 people, with a technical owner and a recurring non-sensitive open-model task. Start with one explicitly supported configuration, preferably already-operational local inference plus Condenser. Native mixed-fleet execution is an extension only after verification.

| Role | What matters | Evidence to collect |
| --- | --- | --- |
| Champion: ML/platform engineer | Fewer manual steps, repeatability, existing-tool fit | Last failed/expensive run, time spent, configuration and task rubric |
| Buyer: founder/engineering lead | Useful result, bounded cost, support ownership | Current spend, approval authority, procurement steps and renewal date |
| Operator/security reviewer | Data route, credentials, rollback, access and model rights | Data classification, endpoint list, machine permissions and install policy |

Qualify before a demo: a weekly task, hardware permission, measurable baseline, named owner, two-week evaluation window and paid-assessment budget. Use public/synthetic data initially. Disqualify production-critical SLAs, regulated data, unsupported hardware, buyers satisfied by a simpler free tool and people interested only in hosting income.

**Secondary segment:** one research lab with an accountable PI/IT administrator, permitted machines and a specific experiment. Student interest is distribution, not procurement authority. University permissions and budgets can prolong sales; do not assume campus machines are available for rental.

**Later:** MSPs serving non-regulated engineering offices after repeatable deployments. Regulated offices require separately demonstrated controls and contracts. Do not launch four verticals at once.

## 4. Packaging and price hypotheses

| Proposed offer | Price to test | Scope and value | Prerequisite |
| --- | --- | --- | --- |
| Community | $0 software license fee | Self-hosted project, docs and community discussion; no service SLA | Accurate status, complete license/release materials |
| Founding team pilot | **$1,500 one-time**, two weeks | Up to 12 delivery hours; one workload/setup; baseline, provider configuration, reproducibility report, runbook and go/no-go recommendation | Signed scope, supported execution path, approved data, rollback and acceptance criteria |
| Team operations/support | **$299/month**, cancellable monthly | One approved deployment, monthly cost/reliability review, maintenance guidance and up to one support hour; extra work quoted | Successful pilot, recurrent need, support capacity and defined response window |
| Future managed team software | Test separately, including the old $199 anchor | Verified team controls with low support demand; no bundled compute | Repeated requests and renewals for software value |
| Future burst/permissioned batch | Separate quote | Measured compute with defined output and full costs | Live integrations, supply, correctness, billing, refunds and reserves |

The pilot does not promise new GPU kernels in twelve hours. It may conclude another tool is best. Charge for agreed deliverables; define cancellation and any refund terms. Obtain separate consent for a case study.

Current offer exclusions: cloud credits, unlimited support, on-call coverage, automatic multi-cloud routing, custom certifications and broad fleet compatibility. These are proposed contract boundaries, not evidence that the service is already sold. Selecting a provider in Condenser does not mean Kadmium procures its compute.

Test prices through comparable scopes and recorded objections. If customers only pay for setup, operate a service business until recurring value is proven; do not call one-off revenue SaaS ARR.

## 5. Value and defensibility

The entry point is **a trusted decision and a repeatable setup**. Common runbooks become installation tooling; repeat incidents become observability/scheduling improvements; recurrent team requirements become paid management features. Prioritize native execution when it demonstrably enables work that a simpler setup cannot.

Potential defensibility comes from opt-in reproducible configurations, failure handling, a maintained support matrix and the operator experience. Do not collect prompts/customer code to build this corpus. Open interfaces and exportable configuration reduce adoption risk. Competitors can copy calculators and UI; reliability across an awkward installed fleet is harder.

For each pilot:

```text
customer benefit = avoided operator hours × customer-approved hourly value
                 + avoided comparable compute/equipment cost
                 - added electricity, networking, administration and migration
net benefit      = customer benefit - Kadmium fee
```

Use the buyer's numbers and quality bar. Exclude speculative revenue gains and double-counted time. If fewer tokens mean worse quality/latency, disclose it. Aggregate RAM and theoretical bandwidth alone do not establish ROI.

## 6. Economic discipline

The [executable model](../scripts/business_model.py) reads [assumptions](business/assumptions.json) and generates the [scenario report](business/ECONOMICS.md). It is independent of runtime constants and makes no network calls. Counts, labor, utilization, reserves and renewals are hypotheses.

At the reference inputs, a twelve-hour pilot contributes **$526.20** before acquisition/fixed costs. One monthly support hour leaves **$194.05** contribution on $299 revenue; another hour removes $75. These are service economics, not 90%+ software margins.

The planning case assumes **six pilots over three months**, with three recurring accounts in month three: **$897 recurring run-rate**. Cohorts and renewal timing are explicit in the model. This is much smaller than the superseded $35,903 monthly cash-billings case and still loses money after acquisition and development. It tests whether to invest, not whether profitability can be announced.

Do not finance operations with expected grants, unsold supply, refundable deposits, unused credits or unproven breakage. The model includes a no-sales case; actual cash is unknown. Replace example costs with real obligations in a weekly 13-week cash forecast. Proposed minimum: three months of essential unrestricted operating cash; stop discretionary spend below it.

### Compute margins and credits

- **25% markup is 20% gross margin:** revenue = cost × 1.25; cost/revenue = 0.80. Reserve against fulfillment/refund obligations; 75% is insufficient for fully redeemed wholesale packs.
- A $29 plan with $35 credits leaves **−$0.14** after full-redemption wholesale cost and a standard US domestic-card fee, before support. The model adds service/refund downside. Do not launch this bundle.
- Prepaid collections are customer liabilities; recognize delivered services under an accountant-reviewed policy. Keep face-value balances separate from fulfillment reserves. Do not count the same pack as both new cash and consumed revenue without a reconciliation.
- Separate purchased, promotional and host-earned credits, terms and cohorts. Circulating host credits are not cash-free fulfillment. Stress full redemption until real cohort history exists.
- Future resale quotes need expiry, price floors, customer-confirmed overages, retry limits and cost ceilings. Include storage, network, idle reservation, interruption, payment, verification, support and refunds. A daily index cannot guarantee margin.

### Proposed marketplace charter

Retain **75% host / 20% operations / 5% community** as a design objective for eligible delivered Supply Grid work, not established legal structure or an automatic company-wide distribution. Define the base as settled eligible compute charges after separately collected tax and refunded work; reconcile subsequent chargebacks. Fees/retries/insurance/overhead must fit the operations allocation.

Illustration: $100 eligible work leaves $20 operations after $75 hosts and $5 community. A $3.20 card fee, $5 verification/retry cost and $2 expected dispute loss leave **$9.80** before support/fixed expenses. Do not stack commissions/royalties on gross billings without recalculating. If economics fail, reprice or decline work rather than silently reducing agreed host shares or spending reserves.

Software/service revenue is separate. Binding stewardship, compensation caps, reserve requirements, grants, golden shares or public-benefit status need governing documents. Describe them as proposals until adopted. Record any future community allocation transparently; additional surplus grants follow obligations, tax and the chosen runway floor.

## 7. Product and commercial gates

| Gate | Required evidence | What becomes appropriate |
| --- | --- | --- |
| G0: credible evaluation | Correct public claims; supported install/rollback; provider/data path; pilot scope; buyer/baseline | Bounded assessments, not managed compute subscriptions |
| G1: paid validation | 3 paid pilots; 2 accepted reports; 2 requests for continuing help on the same problem; logged hours | Standardize the recurring service offer within capacity |
| G2: repeatability | 3 retained accounts for 2 completed billing periods; 2 consented case studies; recurring contribution ≥60%; median support ≤1 hour/account/month | Expand the proven channel; build one common paid operational feature |
| G3: native proof | Committed harness, quality parity, matched single-node baseline, actual multi-device run, supported matrix, power and failure/rollback evidence | Only the measured claims for that configuration |
| G4: compute service | Funded repeat demand, live supply/invoices, accepted-work economics, billing reconciliation, disputes and reserves | Small permissioned compute pilot; broad host recruitment is separate |
| G5: channel/regulated | Repeatable installation/escalation for MSPs; deployment-specific controls and contracts for regulated use | Scoped partner/vertical offers |

Before G1, continuing service is limited to an explicitly requested, month-to-month extension of a successful pilot on the same supported configuration. It still needs a scoped recurring job and available support capacity. The model counts these individual extensions as recurring service; their existence does not establish a repeatable offer or pass G1/G2.

Thresholds are proposed decisions, not statistical proof of product-market fit. Dates are learning windows; gates control promises. Late evidence calls for reduced scope, not relabelling simulations as delivery.

## 8. Capital allocation and options

Focus the next cycle on one workload, reliability, evidence and support tooling. Defer the simultaneous hardware, credit-marketplace, compliance-vertical and global-relay launches.

| Option | Reconsider when | Costs/risks to cover |
| --- | --- | --- |
| Native multi-device runtime | Repeated buyer memory/placement constraint and verified backend | Engineering, correctness, maintained support matrix |
| Private overnight fleet | Funded batch queue and permitted machines | Idle power, retries, verification, compensation, utilization |
| MSP channel | Three repeatable direct deployments and escalation docs | Enablement, support boundaries, discounts |
| Reference hardware | Demand specifically requires an appliance | Landed cost, freight/duties, certification applicability, inventory, returns, model rights |
| Research grant | Technical uncertainty matches current solicitation | Eligibility, PI time, institution agreement, restricted-fund accounting |
| Crowdfunding/equity | Reliable deliverable and a validated capital need | Fulfillment/refunds or securities requirements |

NSF Phase I requires a Project Pitch invitation and eligibility, not merely a university contact. [Source S21](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s21). Cap initial grant-fit research at four founder hours; no award is assumed. Keep hardware and speculative credit liabilities out of the first 90-day budget.

## 9. Commitments worth keeping

Respect owner permissions. Maintain an open self-hostable core with clear license/dependency/model rights. Explain local/external data paths. Never require a speculative token. Report host costs and the possibility of zero demand before earnings. Publish measurements with provenance/failures. Preserve export/uninstall. Make community benefits specific and funded.

The next decision is whether paid users repeatedly value the narrow workflow. The [operating plan](OPERATING_PLAN.md) assigns who gathers that evidence and how to respond if they do not.

