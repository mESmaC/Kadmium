"""Accounting and cohort invariants for the planning model; no runtime dependency."""

import copy
import json
import unittest

import business_model as model


class BusinessModelTests(unittest.TestCase):
    def setUp(self):
        self.inputs = json.loads(model.DEFAULT_INPUT.read_text(encoding="utf-8"))
        self.payment = {"rate": "0.029", "fixed": "0.30"}

    def test_markup_is_not_gross_margin(self):
        self.assertEqual(model.fulfillment_ratio("0.25"), model.dec("0.8"))
        self.assertEqual(1 - model.fulfillment_ratio("0.25"), model.dec("0.2"))

    def test_card_fee_uses_transaction_rounding_and_no_charge_has_no_fee(self):
        self.assertEqual(model.payment_fee("25", self.payment), model.dec("1.03"))
        self.assertEqual(model.payment_fee("0", self.payment), 0)

    def test_partial_redemption_reconciles_liability_and_cash(self):
        result = model.prepaid_bridge("100", "40", "0.25", self.payment)
        self.assertEqual(result["recognized"], 40)
        self.assertEqual(result["deferred"], 60)
        self.assertEqual(result["cogs"], 32)
        self.assertEqual(result["remaining_cash"], model.dec("64.80"))
        self.assertEqual(result["available"], model.dec("4.80"))
        self.assertEqual(result["collected"], result["recognized"] + result["deferred"])
        self.assertEqual(result["remaining_cash"], result["available"] + result["refund_reserve"])

    def test_unredeemed_cash_is_not_revenue_and_full_redemption_releases_liability(self):
        unused = model.prepaid_bridge("100", "0", "0.25", self.payment)
        used = model.prepaid_bridge("100", "100", "0.25", self.payment)
        self.assertEqual(unused["recognized"], 0)
        self.assertEqual(unused["deferred"], 100)
        self.assertEqual(unused["available"], model.dec("-3.20"))
        self.assertEqual(used["deferred"], 0)
        self.assertEqual(used["available"], model.dec("16.80"))
        with self.assertRaises(ValueError):
            model.prepaid_bridge("100", "101", "0.25", self.payment)

    def test_full_credit_redemption_exposes_loss(self):
        result = model.bundle_result(self.inputs, 1)
        self.assertEqual(result["after_compute_and_fee"], model.dec("-0.14"))
        self.assertEqual(result["contribution"], model.dec("-5.72"))
        with self.assertRaises(ValueError):
            model.bundle_result(self.inputs, "1.01")

    def test_cohorts_do_not_book_three_months_for_late_starts(self):
        result = model.scenario_result(self.inputs, self.inputs["scenarios"][2])
        self.assertEqual(result["active"], [0, 1, 3])
        self.assertEqual(result["account_months"], 4)
        self.assertEqual(result["month3_recurring"], 897)
        self.assertEqual(result["revenue"], 10196)
        self.assertEqual(result["result"], model.dec("-17891.60"))

    def test_cannot_convert_same_month_or_convert_a_pilot_twice(self):
        scenario = copy.deepcopy(self.inputs["scenarios"][2])
        scenario["new_recurring_accounts"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            model.scenario_result(self.inputs, scenario)
        scenario["new_recurring_accounts"] = [0, 1, 3]
        with self.assertRaises(ValueError):
            model.scenario_result(self.inputs, scenario)

    def test_no_sales_has_cost_and_unknown_cac(self):
        result = model.scenario_result(self.inputs, self.inputs["scenarios"][0])
        self.assertEqual(result["revenue"], 0)
        self.assertEqual(result["result"], -19050)
        self.assertIsNone(result["pilot_cac"])
        self.assertIsNone(result["retained_cac"])

    def test_more_support_reduces_contribution_and_flags_capacity(self):
        scenario = copy.deepcopy(self.inputs["scenarios"][2])
        base = model.scenario_result(self.inputs, scenario)
        scenario["support_hours_per_account_month"] = "2"
        higher = model.scenario_result(self.inputs, scenario)
        self.assertEqual(base["contribution"] - higher["contribution"], 300)
        scenario["pilot_hours"] = "100"
        self.assertFalse(model.scenario_result(self.inputs, scenario)["within_capacity"])


if __name__ == "__main__":
    unittest.main()
