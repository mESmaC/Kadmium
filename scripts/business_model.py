#!/usr/bin/env python3
"""Reproduce proposed business economics without runtime constants or network I/O."""

import argparse
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/business/assumptions.json"
OUTPUT = ROOT / "docs/business"
CENT = Decimal("0.01")


def dec(value):
    return Decimal(str(value))


def money(value):
    return dec(value).quantize(CENT, rounding=ROUND_HALF_UP)


def usd(value):
    value = money(value)
    return ("-$" if value < 0 else "$") + f"{abs(value):,.2f}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payment_fee(amount, payment):
    require(dec(amount) >= 0, "Charge cannot be negative")
    return money(dec(amount) * dec(payment["rate"]) + dec(payment["fixed"])) if dec(amount) else dec(0)


def fulfillment_ratio(markup):
    require(dec(markup) >= 0, "Markup cannot be negative")
    return dec(1) / (dec(1) + dec(markup))


def prepaid_bridge(sold, consumed, markup, payment):
    sold, consumed = dec(sold), dec(consumed)
    require(0 <= consumed <= sold, "Consumption must be within the purchased balance")
    deferred = sold - consumed
    cogs = money(consumed * fulfillment_ratio(markup))
    fee = payment_fee(sold, payment)
    remaining_cash = sold - cogs - fee
    # Assume unused face value is refundable; do not replace that obligation with COGS.
    return {
        "collected": sold, "recognized": consumed, "deferred": deferred,
        "cogs": cogs, "fee": fee, "remaining_cash": remaining_cash,
        "refund_reserve": deferred, "available": remaining_cash - deferred,
    }


def bundle_result(inputs, redemption):
    bundle, service = inputs["legacy_credit_bundle"], inputs["service"]
    redemption = dec(redemption)
    require(0 <= redemption <= 1, "Redemption must be between zero and one")
    price = dec(bundle["price"])
    fulfillment = money(
        dec(bundle["included_credit"]) * redemption
        * fulfillment_ratio(bundle["wholesale_markup"])
    )
    fee = payment_fee(price, inputs["payment"])
    refund = money(price * dec(service["refund_loss_rate"]))
    contribution = (
        price - fulfillment - fee - refund
        - dec(bundle["monthly_support_cost"]) - dec(bundle["monthly_infra_cost"])
    )
    return {
        "redemption": redemption, "fulfillment": fulfillment,
        "after_compute_and_fee": price - fulfillment - fee,
        "contribution": contribution,
    }


def validate(inputs):
    require(inputs["currency"] == "USD", "This model formats USD only")
    require(0 <= dec(inputs["payment"]["rate"]) < 1, "Invalid payment rate")
    require(dec(inputs["payment"]["fixed"]) >= 0, "Invalid fixed payment fee")
    for key, value in inputs["service"].items():
        require(dec(value).is_finite() and dec(value) >= 0, f"Invalid service input: {key}")
    require(dec(inputs["service"]["refund_loss_rate"]) <= 1, "Invalid refund/loss rate")
    for key in ("pilot_price", "recurring_price", "labor_hourly", "weeks"):
        require(dec(inputs["service"][key]) > 0, f"{key} must be positive")
    for key in ("price", "included_credit", "monthly_support_cost", "monthly_infra_cost"):
        require(dec(inputs["legacy_credit_bundle"][key]) >= 0, f"Invalid bundle input: {key}")
    names = [s["name"] for s in inputs["scenarios"]]
    require(len(set(names)) == len(names), "Scenario names must be unique")


def scenario_result(inputs, scenario):
    service, payment = inputs["service"], inputs["payment"]
    pilots, starts = scenario["delivered_pilots"], scenario["new_recurring_accounts"]
    require(len(pilots) == len(starts) == 3, "Scenarios must contain three months")
    for count in pilots + starts:
        require(type(count) is int and count >= 0, "Account counts must be nonnegative integers")
    for key in ("pilot_hours", "support_hours_per_account_month", "acquisition_hours", "marketing_cash"):
        require(dec(scenario[key]).is_finite() and dec(scenario[key]) >= 0, f"Invalid {key}")
    active, converted, completed = 0, 0, 0
    active_by_month = []
    for delivered, new_accounts in zip(pilots, starts):
        require(new_accounts + converted <= completed, "Recurring accounts need prior-month pilots")
        active += new_accounts
        converted += new_accounts
        completed += delivered
        active_by_month.append(active)

    pilot_count, account_months = sum(pilots), sum(active_by_month)
    pilot_price, recurring_price = dec(service["pilot_price"]), dec(service["recurring_price"])
    rate, labor = dec(service["refund_loss_rate"]), dec(service["labor_hourly"])
    pilot_unit = (
        pilot_price - payment_fee(pilot_price, payment) - money(pilot_price * rate)
        - money(dec(scenario["pilot_hours"]) * labor)
    )
    recurring_unit = (
        recurring_price - payment_fee(recurring_price, payment) - money(recurring_price * rate)
        - money(dec(scenario["support_hours_per_account_month"]) * labor)
        - dec(service["recurring_infra_monthly"])
    )
    revenue = pilot_count * pilot_price + account_months * recurring_price
    contribution = pilot_count * pilot_unit + account_months * recurring_unit
    acquisition = money(dec(scenario["acquisition_hours"]) * labor) + dec(scenario["marketing_cash"])
    fixed = 3 * (dec(service["fixed_product_admin_monthly"]) + dec(service["fixed_overhead_monthly"]))
    result = contribution - acquisition - fixed
    hours = (
        pilot_count * dec(scenario["pilot_hours"])
        + account_months * dec(scenario["support_hours_per_account_month"])
        + dec(scenario["acquisition_hours"])
    )
    capacity = dec(service["commercial_hours_per_week"]) * dec(service["weeks"])
    return {
        "name": scenario["name"], "pilots": pilot_count, "active": active_by_month,
        "account_months": account_months, "month3_recurring": active_by_month[-1] * recurring_price,
        "revenue": revenue, "contribution": contribution, "acquisition": acquisition,
        "fixed": fixed, "result": result, "hours": hours, "capacity": capacity,
        "within_capacity": hours <= capacity,
        "closing_cash_proxy": dec(service["illustrative_opening_cash"]) + result,
        "pilot_unit": pilot_unit, "recurring_unit": recurring_unit,
        "pilot_cac": acquisition / pilot_count if pilot_count else None,
        "retained_cac": acquisition / active_by_month[-1] if active_by_month[-1] else None,
    }


def render_report(inputs, scenarios, bundles, bridge):
    service, bundle, payment = inputs["service"], inputs["legacy_credit_bundle"], inputs["payment"]
    ratio = fulfillment_ratio(bundle["wholesale_markup"])
    lines = [
        "# Illustrative business economics", "",
        f"Generated from [assumptions.json](assumptions.json), dated {inputs['as_of']}.",
        "**Proposals and sensitivities, not a forecast or actual company results.**", "",
        "Regenerate: python scripts/business_model.py --write", "",
        "Verify checked-in outputs: python scripts/business_model.py --check", "",
        "Uses Python's standard library and decimal, transaction-level half-up rounding. "
        "No live rates, runtime fixture imports, external packages or network calls.", "",
        "## 90-day service scenarios", "",
        "All delivered pilots and monthly service are collected in the same month. "
        "Only completed services enter the revenue row. No compute packs, host income, "
        "annual prepayments, hardware, grants or sales taxes are included. Later starts "
        "do not produce three months of recurring revenue.", "",
        "| Metric | " + " | ".join(s["name"] for s in scenarios) + " |",
        "| --- | " + " | ".join("---:" for _ in scenarios) + " |",
    ]
    metrics = [
        ("Completed paid pilots", "pilots", str),
        ("Active recurring accounts, months 1 / 2 / 3", "active", lambda v: " / ".join(map(str, v))),
        ("Recurring account-months", "account_months", str),
        ("Month-three recurring run-rate", "month3_recurring", usd),
        ("90-day delivered-service revenue", "revenue", usd),
        ("Contribution after delivery, support, infra, fees and loss allowance", "contribution", usd),
        ("Acquisition cash + loaded acquisition labor", "acquisition", usd),
        ("Fixed product/admin + overhead", "fixed", usd),
        ("90-day operating result (planning measure)", "result", usd),
        ("Illustrative ending cash proxy", "closing_cash_proxy", usd),
        ("Commercial hours (delivery + support + acquisition)", "hours", lambda v: f"{v:g}"),
        ("Within modeled commercial-hour capacity", "within_capacity", lambda v: "Yes" if v else "NO — reduce starts"),
        ("Loaded CAC per paid pilot", "pilot_cac", lambda v: usd(v) if v is not None else "Not estimable"),
        ("Loaded CAC per month-three recurring account", "retained_cac", lambda v: usd(v) if v is not None else "Not estimable"),
    ]
    for label, key, formatter in metrics:
        lines.append(f"| {label} | " + " | ".join(formatter(s[key]) for s in scenarios) + " |")
    lines += [
        "",
        f"Includes a {dec(service['refund_loss_rate']):.0%} revenue-based refund/loss allowance "
        "as a planning cost, not an accountant's recognition conclusion or funded bank account. "
        "Fixed product/admin labor is separate from direct delivery/acquisition labor; classify "
        "actual costs once.", "",
        f"The cash proxy starts with an illustrative {usd(service['illustrative_opening_cash'])} "
        "and assumes all labor and loss allowances are cash-funded immediately. Actual company "
        "liquidity is unknown. This is not the weekly cash forecast: stress collection delays, tax, "
        "exceptional disputes, payment-product fees and new commitments before spending.", "",
        f"Three months of modeled fixed product/admin and overhead alone equal "
        f"{usd(3 * (dec(service['fixed_product_admin_monthly']) + dec(service['fixed_overhead_monthly'])))}. "
        "These scenarios expose the unmitigated funding need; they do not automatically "
        "apply the operating plan's spending stops. Classify actual essential obligations "
        "and pause discretionary spending before the chosen cash-reserve floor is breached.", "",
        "Short-window scenarios assume no churn after their recurring starts; no LTV is inferred. "
        "The default planning cohort has only one account with a second recurring period by "
        "month three and does not satisfy the three-account, two-period retention gate.", "",
        "## Service contribution and support sensitivity", "",
        "| Scenario cost assumption | Contribution per pilot | Contribution per account-month |",
        "| --- | ---: | ---: |",
    ]
    for s in scenarios:
        lines.append(f"| {s['name']} | {usd(s['pilot_unit'])} | {usd(s['recurring_unit'])} |")
    lines += [
        "",
        f"Each additional support hour removes {usd(service['labor_hourly'])} of contribution. "
        "Prices are tests, not validated willingness to pay. Acquisition/fixed costs are excluded "
        "from these unit contributions and included in the scenario totals.", "",
        f"## Credit-bundle stress: {usd(bundle['price'])} price, {usd(bundle['included_credit'])} credits", "",
        f"At {dec(bundle['wholesale_markup']):.0%} markup, cost is {ratio:.0%} of consumed retail "
        f"credits. Card fee: {dec(payment['rate']):.1%} + {usd(payment['fixed'])}. Additional inputs: "
        f"{usd(bundle['monthly_support_cost'])} support, {usd(bundle['monthly_infra_cost'])} infrastructure "
        f"and {dec(service['refund_loss_rate']):.0%} refund/loss allowance.", "",
        "| Credit redemption | Wholesale fulfillment | After compute + card fee only | After all listed variable costs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for b in bundles:
        lines.append(
            f"| {b['redemption']:.0%} | {usd(b['fulfillment'])} | "
            f"{usd(b['after_compute_and_fee'])} | {usd(b['contribution'])} |"
        )
    lines += [
        "",
        "These are eventual-redemption sensitivities for one billed plan-month, not "
        "recognized margin or spendable cash while credits remain usable. Unredeemed "
        "balances may remain liabilities; no breakage assumption is validated. "
        "The default 52% point is inherited, not an observed cohort. Host-earned credits "
        "do not eliminate fulfillment obligations. See [the chart](ECONOMICS.html).", "",
        "## Prepaid cash versus revenue: partial redemption", "",
        "| Item | Amount |", "| --- | ---: |",
    ]
    for label, key in [
        ("Cash collected", "collected"), ("Delivered compute / recognized revenue", "recognized"),
        ("Unconsumed customer liability", "deferred"), ("Consumed compute cost", "cogs"),
        ("Collection card fee", "fee"), ("Cash remaining", "remaining_cash"),
        ("Reserve if all unused credit is refundable", "refund_reserve"),
        ("Cash above that reserve, before other costs", "available"),
    ]:
        lines.append(f"| {label} | {usd(bridge[key])} |")
    lines += [
        "",
        "Cash collected = delivered revenue + unconsumed liability. Reserves restrict available "
        "cash; they are not another expense or permission to recognize unused balances. Actual "
        "contracts and accounting policy determine recognition, refunds and expiry.", "",
        "## Scope and provenance", "",
        "Payment source: [Stripe US pricing](https://stripe.com/en-us/pricing), checked 2026-10-30. "
        "All other inputs are proposed assumptions. See the "
        "[research register](../MARKETING_BUSINESS_REVIEW_2026-10-30.md#s10) and "
        "[operating plan](../OPERATING_PLAN.md).", "",
    ]
    lines.extend("- " + note for note in inputs["notes"])
    return "\n".join(lines) + "\n"


def render_chart(inputs, bundles):
    bundle, payment = inputs["legacy_credit_bundle"], inputs["payment"]
    maximum = max([abs(float(b["contribution"])) for b in bundles] + [1])
    scale = 90 / maximum
    bars, labels = [], []
    spacing = 600 / max(len(bundles), 1)
    for index, b in enumerate(bundles):
        x = 25 + spacing * (index + 0.5)
        value = float(b["contribution"])
        height = abs(value) * scale
        y = 130 - height if value >= 0 else 130
        color = "var(--chart-1, CanvasText)" if value >= 0 else "var(--destructive, CanvasText)"
        label_y = y - 9 if value >= 0 else y + height + 19
        bars.append(
            f'<rect x="{x - 40}" y="{y:.2f}" width="80" height="{height:.2f}" rx="4" fill="{color}"/>'
            f'<text x="{x}" y="{label_y:.2f}" text-anchor="middle">{html.escape(usd(b["contribution"]))}</text>'
            f'<text x="{x}" y="260" text-anchor="middle">{b["redemption"]:.0%} redeemed</text>'
        )
        labels.append(f'{b["redemption"]:.0%} redemption: {usd(b["contribution"])}')
    full = bundle_result(inputs, 1)
    title = "Credit redemption changes the subscription margin"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kadmium credit-bundle sensitivity</title>
<style>
:root {{ font-family:var(--font-sans,system-ui); color:var(--foreground,CanvasText); }}
body {{ margin:0; }} h2 {{ font-size:20px; margin:0 0 10px; }}
p {{ line-height:1.5; margin:8px 0; }} .muted {{ color:var(--muted-foreground,GrayText); font-size:13px; }}
svg {{ width:100%; height:275px; overflow:visible; }} text {{ fill:var(--foreground,CanvasText); font:15px var(--font-sans,system-ui); }}
.note {{ padding:16px; border-radius:var(--radius,8px); background:var(--secondary,ButtonFace); }}
@media(max-width:420px) {{ svg {{ height:220px; }} }}
</style></head><body>
<h2>{title}</h2>
<p>Eventual contribution per billed month: {usd(bundle['price'])} subscription with {usd(bundle['included_credit'])} compute credits.</p>
<svg viewBox="0 0 650 280" role="img" aria-label="{html.escape('; '.join(labels))}">
<line x1="35" y1="130" x2="625" y2="130" stroke="var(--border,GrayText)"/>
<text x="2" y="135">$0</text>{''.join(bars)}</svg>
<div class="note"><strong>Proposal: separate compute from service pricing.</strong>
<p>Full redemption leaves {usd(full['after_compute_and_fee'])} after wholesale compute and card fees,
before support, infrastructure and the refund/loss allowance.</p></div>
<p class="muted">Illustrative sensitivity, {html.escape(inputs['as_of'])}; not customer results.
Assumes {fulfillment_ratio(bundle['wholesale_markup']):.0%} fulfillment cost,
card fee {dec(payment['rate']):.1%} + {usd(payment['fixed'])},
{usd(bundle['monthly_support_cost'])} support, {usd(bundle['monthly_infra_cost'])} infrastructure
and {dec(inputs['service']['refund_loss_rate']):.0%} refund/loss allowance.
Generated from docs/business/assumptions.json. Eventual redemption rates are unvalidated;
unused balances require separate liability and reserve tracking.</p>
</body></html>
"""


def outputs(inputs):
    validate(inputs)
    scenarios = [scenario_result(inputs, s) for s in inputs["scenarios"]]
    bundles = [bundle_result(inputs, r) for r in inputs["legacy_credit_bundle"]["redemption_rates"]]
    p = inputs["prepaid_example"]
    bridge = prepaid_bridge(p["sold"], p["consumed"], inputs["legacy_credit_bundle"]["wholesale_markup"], inputs["payment"])
    return {
        "ECONOMICS.md": render_report(inputs, scenarios, bundles, bridge),
        "ECONOMICS.html": render_chart(inputs, bundles),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Regenerate checked-in Markdown and chart")
    mode.add_argument("--check", action="store_true", help="Fail if checked-in outputs are stale")
    args = parser.parse_args()
    with DEFAULT_INPUT.open(encoding="utf-8") as handle:
        rendered = outputs(json.load(handle))
    if args.write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, body in rendered.items():
            (OUTPUT / name).write_text(body, encoding="utf-8", newline="\n")
        print("Generated docs/business/ECONOMICS.md and ECONOMICS.html")
    elif args.check:
        stale = [name for name, body in rendered.items() if not (OUTPUT / name).exists()
                 or (OUTPUT / name).read_text(encoding="utf-8") != body]
        if stale:
            raise SystemExit("Stale generated outputs: " + ", ".join(stale))
        print("Business model outputs match assumptions and calculations.")
    else:
        print(rendered["ECONOMICS.md"], end="")


if __name__ == "__main__":
    main()
