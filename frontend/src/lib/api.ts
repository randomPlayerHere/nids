// Thin client for the NIDS FastAPI backend.
//
// Base URL is configurable at build/runtime via VITE_API_BASE. In production
// behind the bundled nginx reverse proxy this is left empty, so requests go to
// the same origin (nginx forwards /api, /ws, /health, ... to the backend).
import type { Alert, ShapValue } from './nids-data';
import { enrichGeo } from './nids-data';

const RAW_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

/** Absolute http(s) base for REST calls. Empty string => same origin. */
export const API_BASE = RAW_BASE;

/** ws(s):// base derived from API_BASE (or the current page origin). */
export function wsBase(): string {
  const http = API_BASE || window.location.origin;
  return http.replace(/^http/, 'ws');
}

// ---- shape coming off the wire (snake_case-ish, timestamp as ISO string) ----
interface RawShap {
  feature: string;
  value: number;
  raw_input?: number;
  description?: string;
}

interface RawAlert {
  id: string;
  timestamp: string | null;
  type: string;
  severity: Alert['severity'];
  confidence: number;
  srcIP: string;
  dstIP: string;
  protocol: string;
  flowDuration: number;
  fwdPackets: number;
  geo: Alert['geo'];
  shapValues?: RawShap[] | null;
}

function normalizeShap(raw: RawShap[] | null | undefined): ShapValue[] | undefined {
  if (!raw) return undefined;
  return raw.map((s) => ({
    feature: s.feature,
    value: s.value,
    description: s.description ?? `Observed value: ${s.raw_input ?? '—'}`,
  }));
}

/** Turn a backend alert into the front-end {@link Alert} (Date + geo fallback). */
export function normalizeAlert(raw: RawAlert): Alert {
  const alert: Alert = {
    id: raw.id,
    timestamp: raw.timestamp ? new Date(raw.timestamp) : new Date(),
    type: raw.type,
    severity: raw.severity,
    confidence: raw.confidence,
    srcIP: raw.srcIP,
    dstIP: raw.dstIP,
    protocol: raw.protocol,
    flowDuration: raw.flowDuration,
    fwdPackets: raw.fwdPackets,
    geo: raw.geo,
    shapValues: normalizeShap(raw.shapValues),
  };
  // The backend only returns geo when a GeoLite2 DB is mounted; for everything
  // else (and for BENIGN flows) we synthesize a stable point so the map stays
  // useful. Attacks always get a location.
  return enrichGeo(alert);
}

export interface AnalyzeSummary {
  total: number;
  benign: number;
  malicious: number;
  by_class: Record<string, number>;
}

export interface AnalyzeResult {
  alerts: Alert[];
  summary: AnalyzeSummary;
}

/** POST a CSV to /api/analyze and return normalized alerts + summary. */
export async function analyzeCsv(file: File, includeShap: boolean): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append('csv', file);
  form.append('include_shap', String(includeShap));

  const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Analysis failed (${res.status}). ${detail.slice(0, 200)}`);
  }
  const data = (await res.json()) as { alerts: RawAlert[]; summary: AnalyzeSummary };
  return { alerts: data.alerts.map(normalizeAlert), summary: data.summary };
}

export interface Health {
  status: string;
  n_features: number;
  n_classes: number;
  labels: string[];
  shap_background_size: number;
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

/**
 * Fire-and-forget ping used to keep the (free-tier Hugging Face Space) backend
 * from going to sleep. Hits /health, swallows any error, and never throws —
 * callers don't care about the result, only that traffic was generated.
 */
export async function pingKeepAlive(): Promise<void> {
  try {
    await fetch(`${API_BASE}/health`, { method: 'GET', cache: 'no-store' });
  } catch {
    // Offline / Space cold-starting — nothing to do, the next tick retries.
  }
}

export type StreamStatus = 'connecting' | 'open' | 'closed';

export interface AlertStream {
  close: () => void;
}

/**
 * Connect to the live /ws/alerts demo stream. Auto-reconnects with backoff.
 * Returns a handle whose close() stops reconnection and tears down the socket.
 */
export function connectAlertStream(
  onAlert: (alert: Alert) => void,
  onStatus?: (status: StreamStatus) => void,
): AlertStream {
  let socket: WebSocket | null = null;
  let closedByCaller = false;
  let retry = 0;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    onStatus?.('connecting');
    socket = new WebSocket(`${wsBase()}/ws/alerts`);

    socket.onopen = () => {
      retry = 0;
      onStatus?.('open');
    };

    socket.onmessage = (event) => {
      try {
        onAlert(normalizeAlert(JSON.parse(event.data) as RawAlert));
      } catch {
        // ignore malformed frames
      }
    };

    socket.onclose = () => {
      onStatus?.('closed');
      if (closedByCaller) return;
      const delay = Math.min(1000 * 2 ** retry, 15000);
      retry += 1;
      reconnectTimer = setTimeout(open, delay);
    };

    socket.onerror = () => socket?.close();
  };

  open();

  return {
    close: () => {
      closedByCaller = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    },
  };
}
