import { describe, it, expect } from "vitest";
import { wsBase, normalizeAlert } from "@/lib/api";

describe("wsBase", () => {
  it("produces a ws(s):// URL, never http(s)://", () => {
    // Derived from VITE_API_BASE (or the page origin): https -> wss, http -> ws.
    const base = wsBase();
    expect(base).toMatch(/^wss?:\/\//);
    expect(base.startsWith("http")).toBe(false);
  });
});

describe("normalizeAlert", () => {
  const raw = {
    id: "a1",
    timestamp: "2024-01-01T00:00:00.000Z",
    type: "DDoS",
    severity: "critical" as const,
    confidence: 0.97,
    srcIP: "10.0.0.5",
    dstIP: "10.0.0.1",
    protocol: "TCP",
    flowDuration: 1200,
    fwdPackets: 8,
    geo: null,
  };

  it("parses the ISO timestamp into a Date", () => {
    const alert = normalizeAlert(raw);
    expect(alert.timestamp).toBeInstanceOf(Date);
    expect(alert.timestamp.toISOString()).toBe(raw.timestamp);
  });

  it("falls back to now() when the timestamp is null", () => {
    const alert = normalizeAlert({ ...raw, timestamp: null });
    expect(alert.timestamp).toBeInstanceOf(Date);
    expect(Number.isNaN(alert.timestamp.getTime())).toBe(false);
  });

  it("synthesizes a geo location for an attack that arrives without one", () => {
    const alert = normalizeAlert(raw);
    expect(alert.geo).not.toBeNull();
    expect(typeof alert.geo?.lat).toBe("number");
  });

  it("maps SHAP values and fills a default description", () => {
    const alert = normalizeAlert({
      ...raw,
      shapValues: [{ feature: "Flow Duration", value: 0.4, raw_input: 1200 }],
    });
    expect(alert.shapValues).toHaveLength(1);
    expect(alert.shapValues?.[0].description).toContain("1200");
  });
});
