import unittest

from calibrate_toy_reference import localized_uv2_targets


class ToyCalibrationTests(unittest.TestCase):
    def test_reported_shifted_gaussian_target(self):
        result = localized_uv2_targets()
        self.assertAlmostEqual(result["t_E_K"], 0.429569975, places=8)
        self.assertAlmostEqual(result["t2_Var_K"], 0.424610244, places=8)


if __name__ == "__main__":
    unittest.main()
