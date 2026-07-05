export type Severity = 'critical' | 'high' | 'medium' | 'low';

export interface Alert {
  id: string;
  timestamp: Date;
  type: string;
  severity: Severity;
  confidence: number;
  srcIP: string;
  dstIP: string;
  protocol: string;
  flowDuration: number;
  fwdPackets: number;
  geo: { lat: number; lng: number; city: string } | null;
  shapValues?: ShapValue[];
}

export interface ShapValue {
  feature: string;
  value: number;
  description: string;
}

const MAJOR_CITIES = [
  { lat: 40.7, lng: -74.0, city: 'New York' },
  { lat: 51.5, lng: -0.1, city: 'London' },
  { lat: 48.9, lng: 2.3, city: 'Paris' },
  { lat: 35.7, lng: 139.7, city: 'Tokyo' },
  { lat: -33.9, lng: 151.2, city: 'Sydney' },
  { lat: 39.9, lng: 116.4, city: 'Beijing' },
  { lat: 19.1, lng: 72.9, city: 'Mumbai' },
  { lat: 55.7, lng: 37.6, city: 'Moscow' },
  { lat: -23.5, lng: -46.6, city: 'São Paulo' },
  { lat: 37.7, lng: -122.4, city: 'San Francisco' },
  { lat: 1.3, lng: 103.8, city: 'Singapore' },
  { lat: 52.5, lng: 13.4, city: 'Berlin' },
  { lat: 25.2, lng: 55.3, city: 'Dubai' },
  { lat: 34.1, lng: -118.2, city: 'Los Angeles' },
  { lat: 41.9, lng: 12.5, city: 'Rome' },
  { lat: 37.6, lng: 127.0, city: 'Seoul' },
  { lat: -34.6, lng: -58.4, city: 'Buenos Aires' },
  { lat: 30.0, lng: 31.2, city: 'Cairo' },
  { lat: 59.3, lng: 18.1, city: 'Stockholm' },
  { lat: 43.7, lng: -79.4, city: 'Toronto' },
];

const ATTACK_TYPES: { type: string; severity: Severity }[] = [
  { type: 'DDoS', severity: 'critical' },
  { type: 'Infiltration', severity: 'critical' },
  { type: 'SSH-Patator', severity: 'high' },
  { type: 'FTP-Patator', severity: 'high' },
  { type: 'PortScan', severity: 'medium' },
  { type: 'Web Attack', severity: 'medium' },
  { type: 'Botnet', severity: 'medium' },
];

const randomIP = () =>
  `${Math.floor(Math.random() * 223) + 1}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;

const PROTOCOLS = ['TCP', 'UDP', 'ICMP'];

const SHAP_FEATURES = [
  { feature: 'Flow Duration', description: 'Total duration of the network flow in microseconds' },
  { feature: 'Fwd Packets', description: 'Number of packets sent in the forward direction' },
  { feature: 'Bwd Packet Length', description: 'Mean size of packets in the backward direction' },
  { feature: 'Flow IAT Mean', description: 'Mean inter-arrival time between packets' },
  { feature: 'SYN Flag Count', description: 'Number of packets with SYN flag set' },
];

function generateShapValues(severity: Severity): ShapValue[] {
  return SHAP_FEATURES.map(f => ({
    ...f,
    value: severity === 'low'
      ? (Math.random() - 0.7) * 0.5
      : (Math.random() - 0.3) * (severity === 'critical' ? 1.5 : severity === 'high' ? 1.0 : 0.7),
  }));
}

let counter = 0;

export function generateAlert(): Alert {
  const isBenign = Math.random() < 0.6;
  counter++;

  if (isBenign) {
    return {
      id: `alert-${counter}-${Date.now()}`,
      timestamp: new Date(),
      type: 'BENIGN',
      severity: 'low',
      confidence: +(0.85 + Math.random() * 0.149).toFixed(3),
      srcIP: randomIP(),
      dstIP: '10.0.0.' + Math.floor(Math.random() * 254 + 1),
      protocol: PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)],
      flowDuration: Math.floor(Math.random() * 500000),
      fwdPackets: Math.floor(Math.random() * 20) + 1,
      geo: null,
      shapValues: generateShapValues('low'),
    };
  }

  const attack = ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)];
  const city = MAJOR_CITIES[Math.floor(Math.random() * MAJOR_CITIES.length)];
  return {
    id: `alert-${counter}-${Date.now()}`,
    timestamp: new Date(),
    type: attack.type,
    severity: attack.severity,
    confidence: +(0.85 + Math.random() * 0.149).toFixed(3),
    srcIP: randomIP(),
    dstIP: '10.0.0.' + Math.floor(Math.random() * 254 + 1),
    protocol: PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)],
    flowDuration: Math.floor(Math.random() * 2000000),
    fwdPackets: Math.floor(Math.random() * 500) + 10,
    geo: { ...city, lat: city.lat + (Math.random() - 0.5) * 6, lng: city.lng + (Math.random() - 0.5) * 8 },
    shapValues: generateShapValues(attack.severity),
  };
}

export const SEVERITY_COLORS: Record<Severity, string> = {
  critical: '#ff4444',
  high: '#ff8800',
  medium: '#ffcc00',
  low: '#444c56',
};

export function getSeverityLabel(severity: Severity): string {
  return { critical: 'Critical', high: 'High', medium: 'Medium', low: 'Low' }[severity];
}

// Deterministically map a source IP to a city so the same flow always lands in
// the same place. Used to fill in geo when the backend has no GeoIP database
// mounted (real lookups, when available, take precedence in api.ts).
function ipToCity(ip: string) {
  let hash = 0;
  for (let i = 0; i < ip.length; i++) hash = (hash * 31 + ip.charCodeAt(i)) >>> 0;
  const base = MAJOR_CITIES[hash % MAJOR_CITIES.length];
  // small deterministic jitter so co-located flows don't perfectly overlap
  const jitter = ((hash >> 8) % 1000) / 1000 - 0.5;
  return { city: base.city, lat: base.lat + jitter * 4, lng: base.lng + jitter * 6 };
}

/**
 * Ensure an alert has a geo point for the map. Keeps a real backend geo if
 * present; otherwise synthesizes a stable one for attack flows. BENIGN flows
 * stay un-located to keep the map focused on threats.
 */
export function enrichGeo(alert: Alert): Alert {
  if (alert.geo) return alert;
  if (alert.type === 'BENIGN') return alert;
  return { ...alert, geo: ipToCity(alert.srcIP) };
}

export function getRecommendedAction(severity: Severity, type: string): string {
  const actions: Record<string, string> = {
    DDoS: 'Immediately enable rate limiting and activate DDoS mitigation. Consider null-routing affected IPs.',
    Infiltration: 'Isolate the affected segment. Initiate incident response and forensic analysis.',
    'SSH-Patator': 'Block source IP. Enforce key-based SSH auth. Review login attempts.',
    'FTP-Patator': 'Disable FTP or restrict access. Block source. Enforce SFTP.',
    PortScan: 'Monitor source IP. Consider temporary block. Review firewall rules.',
    'Web Attack': 'Enable WAF rules. Review application logs. Patch known vulnerabilities.',
    Botnet: 'Quarantine infected hosts. Update signatures. Scan for C2 communication.',
    BENIGN: 'No action required. Traffic appears normal.',
  };
  return actions[type] || 'Investigate further.';
}
