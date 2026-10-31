# Illustrative business economics

Generated from [assumptions.json](assumptions.json), dated 2026-10-30.
**Proposals and sensitivities, not a forecast or actual company results.**

Regenerate: python scripts/business_model.py --write

Verify checked-in outputs: python scripts/business_model.py --check

Uses Python's standard library and decimal, transaction-level half-up rounding. No live rates, runtime fixture imports, external packages or network calls.

## 90-day service scenarios

All delivered pilots and monthly service are collected in the same month. Only completed services enter the revenue row. No compute packs, host income, annual prepayments, hardware, grants or sales taxes are included. Later starts do not produce three months of recurring revenue.

| Metric | No sales | Downside | Planning | Upside |
| --- | ---: | ---: | ---: | ---: |
| Completed paid pilots | 0 | 2 | 6 | 9 |
| Active recurring accounts, months 1 / 2 / 3 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 1 / 3 | 0 / 2 / 6 |
| Recurring account-months | 0 | 1 | 4 | 8 |
| Month-three recurring run-rate | $0.00 | $299.00 | $897.00 | $1,794.00 |
| 90-day delivered-service revenue | $0.00 | $3,299.00 | $10,196.00 | $15,892.00 |
| Contribution after delivery, support, infra, fees and loss allowance | $0.00 | $571.45 | $3,933.40 | $7,788.20 |
| Acquisition cash + loaded acquisition labor | $1,800.00 | $3,225.00 | $4,575.00 | $5,300.00 |
| Fixed product/admin + overhead | $17,250.00 | $17,250.00 | $17,250.00 | $17,250.00 |
| 90-day operating result (planning measure) | -$19,050.00 | -$19,903.55 | -$17,891.60 | -$14,761.80 |
| Illustrative ending cash proxy | $10,950.00 | $10,096.45 | $12,108.40 | $15,238.20 |
| Commercial hours (delivery + support + acquisition) | 20 | 69 | 121 | 148.00 |
| Within modeled commercial-hour capacity | Yes | Yes | Yes | Yes |
| Loaded CAC per paid pilot | Not estimable | $1,612.50 | $762.50 | $588.89 |
| Loaded CAC per month-three recurring account | Not estimable | $3,225.00 | $1,525.00 | $883.33 |

Includes a 2% revenue-based refund/loss allowance as a planning cost, not an accountant's recognition conclusion or funded bank account. Fixed product/admin labor is separate from direct delivery/acquisition labor; classify actual costs once.

The cash proxy starts with an illustrative $30,000.00 and assumes all labor and loss allowances are cash-funded immediately. Actual company liquidity is unknown. This is not the weekly cash forecast: stress collection delays, tax, exceptional disputes, payment-product fees and new commitments before spending.

Three months of modeled fixed product/admin and overhead alone equal $17,250.00. These scenarios expose the unmitigated funding need; they do not automatically apply the operating plan's spending stops. Classify actual essential obligations and pause discretionary spending before the chosen cash-reserve floor is breached.

Short-window scenarios assume no churn after their recurring starts; no LTV is inferred. The default planning cohort has only one account with a second recurring period by month three and does not satisfy the three-account, two-period retention gate.

## Service contribution and support sensitivity

| Scenario cost assumption | Contribution per pilot | Contribution per account-month |
| --- | ---: | ---: |
| No sales | $526.20 | $194.05 |
| Downside | $226.20 | $119.05 |
| Planning | $526.20 | $194.05 |
| Upside | $676.20 | $212.80 |

Each additional support hour removes $75.00 of contribution. Prices are tests, not validated willingness to pay. Acquisition/fixed costs are excluded from these unit contributions and included in the scenario totals.

## Credit-bundle stress: $29.00 price, $35.00 credits

At 25% markup, cost is 80% of consumed retail credits. Card fee: 2.9% + $0.30. Additional inputs: $4.00 support, $1.00 infrastructure and 2% refund/loss allowance.

| Credit redemption | Wholesale fulfillment | After compute + card fee only | After all listed variable costs |
| --- | ---: | ---: | ---: |
| 0% | $0.00 | $27.86 | $22.28 |
| 52% | $14.56 | $13.30 | $7.72 |
| 100% | $28.00 | -$0.14 | -$5.72 |

These are eventual-redemption sensitivities for one billed plan-month, not recognized margin or spendable cash while credits remain usable. Unredeemed balances may remain liabilities; no breakage assumption is validated. The default 52% point is inherited, not an observed cohort. Host-earned credits do not eliminate fulfillment obligations. See [the chart](ECONOMICS.html).

## Prepaid cash versus revenue: partial redemption

| Item | Amount |
| --- | ---: |
| Cash collected | $100.00 |
| Delivered compute / recognized revenue | $40.00 |
| Unconsumed customer liability | $60.00 |
| Consumed compute cost | $32.00 |
| Collection card fee | $3.20 |
| Cash remaining | $64.80 |
| Reserve if all unused credit is refundable | $60.00 |
| Cash above that reserve, before other costs | $4.80 |

Cash collected = delivered revenue + unconsumed liability. Reserves restrict available cash; they are not another expense or permission to recognize unused balances. Actual contracts and accounting policy determine recognition, refunds and expiry.

## Scope and provenance

Payment source: [Stripe US pricing](https://stripe.com/en-us/pricing), checked 2026-10-30. All other inputs are proposed assumptions. See the [research register](../MARKETING_BUSINESS_REVIEW_2026-10-30.md#s10) and [operating plan](../OPERATING_PLAN.md).

- Prices and all non-payment inputs are hypotheses, not vendor or customer facts.
- Recurring accounts begin no earlier than the month after pilot delivery. No later churn is assumed inside the short model; do not infer lifetime retention.
- Recurring service includes individually scoped pilot extensions. Standardizing the offer and expanding acquisition remain subject to G1/G2 in the business plan.
- All modeled services are delivered, invoiced and collected within the month. No unpaid receivables or deferred service balances at month end; stress collection delay separately.
- Direct delivery and acquisition hours are additional paid/loaded labor capacity, not hours also charged to fixed product/admin expense.
- Operating result is a planning contribution after risk allowances, acquisition and fixed costs, not GAAP profit. Taxes, financing, exceptional disputes, custom legal work and hardware are excluded.
- Cash proxy assumes all modeled labor and risk allowances are cash-funded in the period. Opening cash is illustrative and is not a statement of actual company funds.
- First 90 days exclude cloud resale, host payouts, grants, hardware and annual prepayments.
- Dollar amounts are rounded at the per-transaction line level using decimal half-up rounding.
