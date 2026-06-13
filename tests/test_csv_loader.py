import io
import unittest

from fastapi import HTTPException, UploadFile

from scripts.app.services.csv_loader import load_csv, synth_meta
from scripts.app.services.model import FEATURE_NAMES


def _upload(text: str) -> UploadFile:
    return UploadFile(filename="test.csv", file=io.BytesIO(text.encode()))


def _csv(rows: list[list[str]]) -> str:
    header = ",".join(FEATURE_NAMES)
    body = "\n".join(",".join(r) for r in rows)
    return header + "\n" + body


class TestCsvLoader(unittest.TestCase):
    def test_loads_rows(self):
        out = load_csv(_upload(_csv([["0"] * len(FEATURE_NAMES), ["1"] * len(FEATURE_NAMES)])))
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out[0].keys()), set(FEATURE_NAMES))

    def test_inf_and_nan_become_zero(self):
        bad = ["inf"] + ["nan"] * (len(FEATURE_NAMES) - 1)
        out = load_csv(_upload(_csv([bad])))
        self.assertTrue(all(v == 0.0 for v in out[0].values()))

    def test_missing_columns_raise(self):
        with self.assertRaises(HTTPException):
            load_csv(_upload("Flow Duration,Total Fwd Packets\n1,2\n"))

    def test_empty_raises(self):
        with self.assertRaises(HTTPException):
            load_csv(_upload(",".join(FEATURE_NAMES) + "\n"))

    def test_synth_meta(self):
        row = {name: 0.0 for name in FEATURE_NAMES}
        row["Flow Duration"] = 12345.0
        row["Total Fwd Packets"] = 7.0
        meta = synth_meta(row)
        self.assertEqual(meta.flow_duration, 12345)
        self.assertEqual(meta.fwd_packets, 7)
        self.assertIn(meta.protocol, {"TCP", "UDP", "ICMP"})


if __name__ == "__main__":
    unittest.main()
