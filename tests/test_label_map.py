import unittest

from scripts.app.services.label_map import to_severity


class TestLabelMap(unittest.TestCase):
    def test_known_attacks(self):
        self.assertEqual(to_severity("DDoS"), "critical")
        self.assertEqual(to_severity("SSH-Patator"), "high")
        self.assertEqual(to_severity("PortScan"), "medium")
        self.assertEqual(to_severity("BENIGN"), "low")

    def test_unknown_defaults_to_low(self):
        self.assertEqual(to_severity("Something New"), "low")


if __name__ == "__main__":
    unittest.main()
