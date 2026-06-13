import importlib.util
import unittest

# these need the full stack (tensorflow + httpx); skip if not installed
_HAVE_STACK = all(importlib.util.find_spec(m) for m in ("tensorflow", "httpx"))


@unittest.skipUnless(_HAVE_STACK, "needs tensorflow + httpx")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from scripts.api import app

        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()  # triggers lifespan -> loads model

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["n_classes"], 11)

    def test_features(self):
        self.assertEqual(len(self.client.get("/features").json()["features"]), 78)

    def test_predict(self):
        r = self.client.post("/predict", json={"features": [0.0] * 78})
        self.assertEqual(r.status_code, 200)
        self.assertIn("predicted_class", r.json())

    def test_stream(self):
        with self.client.websocket_connect("/ws/alerts") as ws:
            alert = ws.receive_json()
            self.assertIn("severity", alert)


if __name__ == "__main__":
    unittest.main()
