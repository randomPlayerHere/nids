import { FC, useMemo, useRef } from 'react';
import { Globe as GlobeIcon } from 'lucide-react';
import { World, type GlobeConfig } from '@/components/ui/globe';
import { Alert, SEVERITY_COLORS } from '@/lib/nids-data';

type Arc = {
  order: number;
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  arcAlt: number;
  color: string;
};

interface GlobePanelProps {
  alerts: Alert[];
}

// Where attack arcs converge — a notional SOC / datacenter (Ashburn, VA).
const HQ = { lat: 39.0438, lng: -77.4874 };

// How many recent attacks to draw as arcs (keeps the 3D scene smooth).
const MAX_ARCS = 18;

// Themed to the dashboard's neutral-dark + warm-severity palette (no blue):
// a lifted graphite globe, bright warm-white landmass dots, and an amber
// atmosphere glow that echoes the high/medium severity hues.
const globeConfig: GlobeConfig = {
  pointSize: 1,
  globeColor: '#1c1c20',
  showAtmosphere: true,
  atmosphereColor: '#f0a23c',
  atmosphereAltitude: 0.15,
  emissive: '#1c1c20',
  emissiveIntensity: 0.18,
  shininess: 1.1,
  polygonColor: 'rgba(246,238,226,0.72)',
  ambientLight: '#d8d2c8',
  directionalLeftLight: '#ffffff',
  directionalTopLight: '#ffffff',
  pointLight: '#f0c98a',
  arcTime: 1600,
  arcLength: 0.9,
  rings: 1,
  maxRings: 3,
  autoRotate: true,
  autoRotateSpeed: 0.6,
};

const GlobePanel: FC<GlobePanelProps> = ({ alerts }) => {
  const attacks = useMemo(
    () => alerts.filter(a => a.geo && a.severity !== 'low').slice(0, MAX_ARCS),
    [alerts],
  );

  // Keep one stable arc object per alert id. three-globe tags datums it has
  // already rendered, so reusing the same reference for an existing arc lets it
  // finish its flight animation uninterrupted — only a newly-added arc animates
  // in. Rebuilding fresh objects each tick (the naive approach) restarts every
  // arc and is what made in-flight arcs "break".
  const arcCache = useRef<Map<string, Arc>>(new Map());
  const arcSeq = useRef(0);

  const key = attacks.map(a => a.id).join('|');
  const data = useMemo(() => {
    const cache = arcCache.current;
    const present = new Set(attacks.map(a => a.id));
    for (const id of [...cache.keys()]) {
      if (!present.has(id)) cache.delete(id); // drop arcs that scrolled out of the window
    }
    attacks.forEach(a => {
      if (!cache.has(a.id)) {
        const n = arcSeq.current++;
        cache.set(a.id, {
          order: (n % 8) + 1,
          startLat: a.geo!.lat,
          startLng: a.geo!.lng,
          endLat: HQ.lat,
          endLng: HQ.lng,
          arcAlt: 0.1 + (n % 5) * 0.06,
          color: SEVERITY_COLORS[a.severity],
        });
      }
    });
    return attacks.map(a => cache.get(a.id)!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-background">
      <World globeConfig={globeConfig} data={data} />

      {/* Title overlay */}
      <div className="pointer-events-none absolute top-3 left-3 z-10 flex items-center gap-2 panel-glass border border-border rounded-md px-2.5 py-1.5">
        <GlobeIcon size={11} className="text-muted-foreground" />
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
          Threat Map
        </span>
        <span className="text-[10px] font-mono text-foreground tabular-nums">{attacks.length}</span>
      </div>

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-10 panel-glass border border-border rounded-md px-3 py-2 space-y-1">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Severity</p>
        {[
          { label: 'Critical', color: SEVERITY_COLORS.critical },
          { label: 'High', color: SEVERITY_COLORS.high },
          { label: 'Medium', color: SEVERITY_COLORS.medium },
        ].map(s => (
          <div key={s.label} className="flex items-center gap-2 text-[10px] font-mono">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
            <span className="text-foreground/80">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GlobePanel;
