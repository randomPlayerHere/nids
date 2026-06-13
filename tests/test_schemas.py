import unittest

from scripts.app.schemas import Alert, AnalyzeResponse, AnalyzeSummary, Geo


class TestSchemas(unittest.TestCase):
    def test_alert_shape(self):
        alert = Alert(
            id="a1",
            type="DDoS",
            severity="critical",
            confidence=0.99,
            srcIP="8.8.8.8",
            dstIP="10.0.0.5",
            protocol="TCP",
            flowDuration=1000,
            fwdPackets=42,
            geo=Geo(lat=1.0, lng=2.0, city="Nowhere"),
        )
        dumped = alert.model_dump(mode="json")
        for key in ("srcIP", "dstIP", "flowDuration", "fwdPackets", "shapValues"):
            self.assertIn(key, dumped)
        self.assertEqual(dumped["geo"]["city"], "Nowhere")

    def test_analyze_summary(self):
        summary = AnalyzeSummary(total=3, benign=1, malicious=2, by_class={"BENIGN": 1, "DDoS": 2})
        resp = AnalyzeResponse(alerts=[], summary=summary)
        self.assertEqual(resp.summary.malicious, 2)


if __name__ == "__main__":
    unittest.main()
