# Kadmium operating plan

**Proposed 2026-10-30.** This plan operationalizes the [business strategy](BUSINESS_PLAN_AND_STANCE.md). Roles are responsibilities, not assumed hires. One founder may hold several roles, but each decision and cost needs a named owner. Budgets and legal structures require adoption; this review has not created accounts, sent messages or incurred expenditure.

## 1. Work within real capacity

For a single full-time operator, initially reserve 24 hours/week for product engineering and administration and 16 for discovery, delivery and support. Cap active pilots at **two**, and scope each to twelve delivery hours over two weeks. Stop accepting new starts when promised support plus delivery exceeds available hours. A waiting list is better than hidden unpaid overtime.

The planning model's six pilots require 72 delivery hours plus four account-months of support. Add 45 acquisition hours: 121 commercial hours across thirteen weeks. These are assumed hours, not proof of staffing. Keep a time log and account for context switching. Budget direct delivery/acquisition labor separately from non-delivery product/admin costs; never count the same founder hours twice.

Use one customer task template, one supported deployment path and one report format. Fix the repeated problem before adding another backend or vertical. Require three independent customer requests for a shared feature, unless a correctness/reliability defect makes it essential.

## 2. Ownership and weekly rhythm

| Responsibility | Accountable role | Routine and decision artifact |
| --- | --- | --- |
| ICP, qualification, pricing, renewal | Founder/commercial lead | Monday: pipeline review with source, stage, next action, owner and date; mark stalled deals and losses |
| Supported configuration, release, measurements | Engineering lead | Twice weekly: reproduce the target workflow; review failures and support cost; maintain versioned compatibility record |
| Pilot acceptance and customer continuity | Delivery/support owner | Agreed checkpoints and a short weekly outcome note; maintain runbook and rollback route |
| Cash, balances, unit economics | Finance owner; accountant for accounting/tax policy | Friday: reconcile bank/processor/invoices, liabilities and 13-week forecast; monthly close |
| Public claims, sources, campaign spend | Marketing owner, engineering co-review for technical claims | Before publishing: claim/evidence/date/owner review; monthly competitor refresh |
| Contract, data and entity decisions | Founder with relevant legal/security expertise | Resolve actual deployment requirements before making contractual promises |
| Community allocation/charter | Founder and future governance body | Record proposal/adoption status; publish only real approved allocations, not simulated outcomes |

Friday decision memo, one page: what customers completed; payments/refunds; delivery/support hours; critical failures; cash/runway; experiment results and denominator; one decision to continue, change or stop. Do not substitute GitHub stars, synthetic tokens or connected-node counts for these outcomes.

## 3. Pilot delivery playbook

1. **Qualify and scope.** Confirm the buyer, champion, operator, repeat task, hardware permission, approved data, current baseline and budget. Reject unsupported/regulated/production-critical work for the initial cohort.
2. **Agree a statement of work.** One workload/configuration, twelve-hour limit, milestone deliverables, fee/timing, cancellation/refund terms, customer compute costs, support boundaries and acceptance date. No automatic long-term renewal.
3. **Record the starting condition.** Versions, selected provider/endpoint, data destination, installation access, model/dependency rights, existing cost/quality and rollback plan. Do not collect secrets in marketing/CRM notes.
4. **Set up the supported path.** Customer-owned credentials and provider billing; explicit opt-in for external inference. Condenser compute inspection remains a planning preview. Do not imply it deploys models or switches provider automatically.
5. **Run the agreed task.** A public/synthetic fixture first, then only approved data. Capture successful/failed runs, quality, end-to-end latency, cold/warm state and operator time. Distinguish actual generation from reference diagnostics.
6. **Deliver the report and runbook.** Baseline, comparison, costs, limitations, failure recovery and a recommendation to continue/change/stop. A negative result is a valid assessment if scoped that way.
7. **Acceptance and renewal discussion.** Customer repeats the useful task. Review whether there is a recurring operational job worth the $299 proposal. Before G1, any continuing work is a scoped, month-to-month pilot extension; standardize the offer only after G1. Do not sell subscription value that only existed during setup.
8. **Close or hand over.** Revoke temporary access, export/delete agreed artifacts, return ownership to the operator, record service time and loss reasons. Request separate written consent before naming a customer or publishing data.

### Minimum pilot record

```text
account_id / source / stage / next_action_date / accountable_owner
buyer / champion / operator / hardware_permission
workload / quality_rubric / incumbent / last_incident
supported_versions / model_license / provider_endpoint / approved_data_class
baseline / acceptance_threshold / delivery_hours_cap / rollback
contract_value / invoice_status / payment_date / refund_obligation
actual_delivery_hours / support_hours / direct_cost / accepted_result
renewal_decision_date / recurring_job / loss_reason / publication_consent
```

No API keys, private prompt bodies or customer datasets belong in this record.

## 4. Release and claim management

Maintain a small **claim register**: exact wording, surface, current status (implemented/reported/modelled/planned), evidence link/commit, allowed scope, owner, review date and expiry trigger. Engineering signs technical evidence; marketing owns the wording. A test of reference kernels cannot approve a production throughput claim.

The initial publication-remediation backlog is concrete:

| ID | Surface / correction | Acceptance evidence | Priority |
| --- | --- | --- | --- |
| C1 | [Marketing site](../crates/kadmium-gateway/src/marketing_site.html): hero, plans, earnings calculator and CTAs | No active offer for unimplemented billing/regulated services; draft copy and prices reconciled with this plan; simulated figures clearly labelled; functioning destinations | P0 before traffic |
| C2 | README and guides: model execution, compatibility, USB4/failover and privacy claims | Each present-tense claim tied to supported execution evidence; planned targets labelled; no contradictory install/generation instructions | P0 before traffic |
| C3 | Benchmark references | Run 1 described as BF16/CPU/external harness with unresolved caveats; no Q4/1GbE/zero-swap substitution; committed matched benchmark before any speed headline | P0 |
| C4 | Companion/dashboard/runtime economic outputs | Clear fixture/simulation boundaries; no implied live quotes, real balances, completed bank settlement or actual customers | P0 before paid compute |
| C5 | License/release package | License files match manifest, dependencies/model rights reviewed; tagged artifacts, checksum/install and rollback verified on each advertised OS | P0 before packaged distribution |
| C6 | Provider/data path | User can identify local/external route; no quiet failover; approved snapshot/tool contents; support exports redacted | P0 before pilot use |
| C7 | Consent and attribution | Customer/creator permission and disclosure; campaign IDs; no invented logos, endorsements or “live” counters | Before campaign |

**Remaining scope:** this research PR does not implement C1–C7. G0 is not passed until applicable blockers are resolved. Engineering execution and public launch are separate decisions; merging a research plan does not publish a compliant product.

### Benchmark acceptance for a commercial claim

Commit the harness and exact code revision; pin/checksum model assets within redistribution rights. Record hardware, actual memory, drivers, OS, link, backend, quantization, input/output lengths, cache, concurrency, prompt set, quality, wall power, total and component timing. Include matched single-node and alternative-runtime baselines. Repeat runs enough to show variation (initial target: at least five complete runs per configuration, disclose sample size). Report medians and tail metrics with failure counts; do not imply robust tail estimates from tiny samples.

Differentiate feasibility from acceleration: fitting a model across two machines can be useful even if slower than one sufficiently large machine. Record the second machine's incremental cost/energy. Test interruption/recovery before describing failover; distinguish prewarmed from cold startup. No shared-latency SLA until operations can sustain it.

## 5. Customer support and reliability

For the founding service, propose business-hours support with an **initial response by the next business day**, not a resolution guarantee or 24/7 uptime SLA. Publish actual hours/timezone in the contract. Let users contact support outside the window without implying on-call staffing.

| Incident | Action | Owner and follow-through |
| --- | --- | --- |
| Suspected data exposure, credential misuse or unexpected paid execution | Stop the affected workflow/provider route; preserve minimum necessary evidence; follow agreed customer notice and credential-rotation process | Engineering/founder; assess scope before resuming |
| A supported workflow is broken | Roll back or use the documented customer-approved alternative; communicate impact/next checkpoint | Support + engineering; prioritize reproduction |
| Unsupported configuration or feature request | State boundary and provide a fit decision; do not silently expand the SOW | Delivery owner; quote separately if appropriate |
| Repeated support overruns | Limit new starts, improve runbook, narrow support matrix or reprice at renewal | Founder; no hidden unlimited service |

Track incidents per accepted workload, time to recover, and support hours per account. A 60% recurring contribution target and one-hour median support limit guide scope; they are not achieved metrics. If response commitments cannot be met, stop acquisition and inform affected customers.

## 6. Finance, treasury and contracting

**Stage one:** bill for assessment and service; the customer pays inference providers directly. Use ordinary invoices with documented acceptance. No host wallet, refundable compute vault, passive-income pool, speculative token or promotional compute obligation is needed to validate the business.

Maintain:
- invoice ledger, accounts receivable and delivery/recognition status;
- bank/processor balances and reconciliation, including fees and chargebacks;
- deferred service revenue and refundable customer balances;
- direct delivery/support expense and loaded acquisition time;
- fixed product/admin expenses, tax payables and contractual commitments;
- grant-restricted funds and future host/community payables separately if they arise.

Refresh a **13-week cash forecast weekly**. Show opening unrestricted cash, collections by realistic date, payroll/contractors, refunds, provider bills, tax, restricted balances, closing cash and minimum reserve. A delayed invoice is not usable cash. Use a no-sales case; funded grants/credits do not renew automatically.

Cash reserve and accounting liability are different. Example: $100 prepaid compute, $40 consumed, $60 unconsumed. At 80% cost, consumed work costs $32. With a $3.20 card fee, remaining cash is $64.80. If $60 is refundable, retain $60 and only $4.80 is unencumbered before other costs. The full $60 customer liability does not shrink to $48 because expected fulfillment is cheaper. The [model](business/ECONOMICS.md) reproduces this bridge.

For a future compute marketplace, validate:
1. Live supplier APIs, contracts, region/availability and resale permissions.
2. Idempotent metering with model/task identity; verified **accepted output** and duplicate/retry handling.
3. Refunds, payment disputes, host identity/payout requirements and prohibited work handling.
4. Reconciliation of customer cash, consumed revenue, credits, host payables and reserves; no invented escrow claim.
5. Price caps, stop-spend limits, stalled-job cutoff and provider shutdown verification.
6. Reserve stress at full redemption and price/cost deterioration, including fees and failed work.

Stripe describes dispute windows commonly far beyond seven days; a T+7 host policy is a timing choice, not elimination of chargebacks. [Source](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s11). A closed-loop label does not decide prepaid-access treatment, state obligations, expiry/refunds or tax. [Source](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s16). Obtain advice on the actual design before operating it; do not use promotional credits as a loophole in the model.

## 7. Data, claims and procurement boundaries

The first cohort avoids PHI/CUI and other regulated workloads. On-premise placement is not by itself legal compliance, attorney-client privilege, isolation or a security assessment. External inference, remote support, logs, backups and tool calls can change the boundary. HHS explicitly rejects treating encryption/no-key access as an automatic exemption for cloud business associates. [Source](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s14).

Describe what is evidenced: selected endpoint, what data is sent, where artifacts remain, access controls and deletion/export. Hidden activations and rotations are not a confidentiality proof. A packet capture showing no plaintext is not proof of non-recoverability.

Before a later regulated pilot, identify actual deployment-specific requirements, contracts, access/isolation, validated cryptography where required, auditability, backups and incident duties. Do not make a universal certification assertion or treat SOC 2 as an automatic substitute for HIPAA/CMMC analysis.

## 8. Partners, mission and optional hardware

MSPs need a repeatable install, a support boundary, escalation window, liability/contract split and a margin after service. Begin with one non-exclusive engineering-office partner. A reseller discount alone is not a channel strategy; compare direct and channel contribution with real hours.

University partnerships need hardware-owner approval, permitted datasets/models, scheduled resource access, ownership/publication terms and a real funding source. A club invitation cannot authorize department compute rental. NSF eligibility/invitations are separate from product interest; no grant is budgeted. [Source](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s21).

Hardware remains deferred. Before reconsideration, cost freight, duties, taxes, safety/EMC applicability, SSDs, image preparation, model redistribution, customer support, returns and warranty. A board's certification does not establish the final system's compliance or a zero-fee launch. A donation needs verified ownership, transfer and data-handling arrangements; no charitable deduction/ESG certification promise without an eligible recipient and actual substantiation.

For environmental reporting, measure incremental wall energy, workload completion and avoided purchase assumptions. Include idle time and displaced cloud/local alternative; distinguish reuse from lower operational energy. Do not promise avoided grid construction or zero water/carbon from a node counter. [FTC/IEA sources](MARKETING_BUSINESS_REVIEW_2026-10-30.md#s15)

## 9. Decision log and stop rules

At day 30: if ten qualified buyers cannot identify a repeat problem they will pay to address, change the segment/offer before adding features.

At day 60: if pilots cannot be delivered within scope or buyers do not request continuing work, narrow configurations, reprice, or remain a one-off service. Pause sales when reliability or support commitments fail.

At day 90: review actual cohorts, renewals, contribution, workload outcomes and runway. Pass G2 only on its evidence. If only the developer tool is valued, keep it open and pursue support/sponsorship aligned to usage; do not invent SaaS retention. If a recurring native-capacity need is demonstrated, fund a defined engineering milestone with those design partners.

For every decision record: date, evidence, alternatives considered, owner, spending cap, expected result, review date and reversal/stop condition. The task is to learn whether this business works while preserving the larger vision and the ability to change course.
