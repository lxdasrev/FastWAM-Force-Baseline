import unittest

from test_force_mask_and_loss import (
    test_all_invalid_force_keys_are_removed_without_empty_attention_rows,
    test_all_invalid_force_loss_is_differentiable_zero,
    test_all_invalid_force_rows_are_zero_and_finite,
    test_force_flow_loss_is_per_sample_scheduler_weighted,
    test_force_mot_mask_matches_v0_vplus_f0_fplus_action_contract,
    test_force_mot_mask_removes_invalid_force_keys_and_has_no_empty_rows,
    test_invalid_force_keys_do_not_change_valid_local_attention_outputs,
    test_two_expert_mot_mask_contract_is_unchanged,
    test_zero_force_gate_exactly_preserves_va_and_gate_gets_gradient,
)
from test_opentouch_retarget import (
    test_action_and_proprio_share_initial_palm_canonical_frame,
    test_action_dimension_seven_is_next_absolute_gripper_target,
)


class FastWAMForceAcceptanceTests(unittest.TestCase):
    def test_canonical_frame(self):
        test_action_and_proprio_share_initial_palm_canonical_frame()

    def test_absolute_gripper_target(self):
        test_action_dimension_seven_is_next_absolute_gripper_target()

    def test_invalid_key_invariance(self):
        test_invalid_force_keys_do_not_change_valid_local_attention_outputs()

    def test_all_invalid_attention(self):
        test_all_invalid_force_rows_are_zero_and_finite()

    def test_mot_mask(self):
        test_force_mot_mask_removes_invalid_force_keys_and_has_no_empty_rows()

    def test_exact_three_expert_mot_mask(self):
        test_force_mot_mask_matches_v0_vplus_f0_fplus_action_contract()

    def test_all_invalid_force_mot_keys(self):
        test_all_invalid_force_keys_are_removed_without_empty_attention_rows()

    def test_two_expert_mot_mask_unchanged(self):
        test_two_expert_mot_mask_contract_is_unchanged()

    def test_weighted_force_loss(self):
        test_force_flow_loss_is_per_sample_scheduler_weighted()

    def test_all_invalid_force_loss(self):
        test_all_invalid_force_loss_is_differentiable_zero()

    def test_zero_force_gate_preserves_va(self):
        test_zero_force_gate_exactly_preserves_va_and_gate_gets_gradient()


if __name__ == "__main__":
    unittest.main()
