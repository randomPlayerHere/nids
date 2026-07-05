import { FC, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Globe, WifiOff } from 'lucide-react';
import { Alert, SEVERITY_COLORS } from '@/lib/nids-data';

interface MapPanelProps {
  alerts: Alert[];
}

const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

const MapPanel: FC<MapPanelProps> = ({ alerts }) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<L.Map | null>(null);
  const markersRef = useRef<Map<string, L.CircleMarker>>(new Map());
  const [offline, setOffline] = useState(false);

  const geoAttacks = useMemo(
    () => alerts.filter(a => a.geo && a.severity !== 'low'),
    [alerts]
  );

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    try {
      const container = mapRef.current;
      const map = L.map(container, {
        zoomControl: false,
        attributionControl: true,
        worldCopyJump: false,
        zoomSnap: 0,
        zoomDelta: 0.01,
        dragging: false,
        scrollWheelZoom: false,
        doubleClickZoom: false,
        boxZoom: false,
        keyboard: false,
        touchZoom: false,
        inertia: false,
      });
      const tiles = L.tileLayer(TILE_URL, {
        attribution: '&copy; CartoDB',
        subdomains: 'abcd',
        maxZoom: 18,
        noWrap: true,
      }).addTo(map);

      // Only flag offline if *no* tile loads in the first few seconds.
      // A single failing tile (edge, transient 404) shouldn't kill the map.
      let loadedAny = false;
      tiles.on('tileload', () => { loadedAny = true; });
      const offlineTimer = window.setTimeout(() => {
        if (!loadedAny) setOffline(true);
      }, 4000);
      (map as L.Map & { _offlineTimer?: number })._offlineTimer = offlineTimer;

      // Pick a fractional zoom so the world's width matches the container's
      // width exactly — no horizontal blank space, and only the polar regions
      // get cropped vertically (which is fine).
      const refit = () => {
        const w = container.offsetWidth;
        if (w <= 0) return;
        const z = Math.log2(w / 256);
        map.invalidateSize({ animate: false });
        map.setView([18, 0], z, { animate: false });
      };
      refit();

      const ro = new ResizeObserver(refit);
      ro.observe(container);
      (map as L.Map & { _ro?: ResizeObserver })._ro = ro;

      mapInstance.current = map;
    } catch {
      setOffline(true);
    }
    return () => {
      const m = mapInstance.current as (L.Map & { _ro?: ResizeObserver; _offlineTimer?: number }) | null;
      m?._ro?.disconnect();
      if (m?._offlineTimer) window.clearTimeout(m._offlineTimer);
      m?.remove();
      mapInstance.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapInstance.current) return;
    const map = mapInstance.current;

    alerts.forEach(alert => {
      if (!alert.geo || alert.severity === 'low') return;
      if (markersRef.current.has(alert.id)) return;

      const color = SEVERITY_COLORS[alert.severity];
      const marker = L.circleMarker([alert.geo.lat, alert.geo.lng], {
        radius: 7,
        color,
        fillColor: color,
        fillOpacity: 0.55,
        weight: 1.5,
        opacity: 1,
      }).addTo(map).bindTooltip(
        `<div style="font-family:JetBrains Mono,monospace;font-size:11px;line-height:1.5">
           <div style="color:${color};font-weight:600">${alert.type}</div>
           <div style="color:#9ca3af">${alert.geo.city} · ${alert.srcIP}</div>
         </div>`,
        { className: 'nids-tooltip', direction: 'top', offset: [0, -8] }
      );

      markersRef.current.set(alert.id, marker);

      setTimeout(() => {
        marker.setStyle({ opacity: 0.25, fillOpacity: 0.25 });
      }, 60000);
    });
  }, [alerts]);

  if (offline) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="mx-auto h-10 w-10 rounded-full border border-border bg-card flex items-center justify-center mb-3">
            <WifiOff size={15} className="text-muted-foreground" />
          </div>
          <p className="text-foreground text-[13px] font-medium">Map tiles unavailable</p>
          <p className="text-muted-foreground/70 text-[11px] mt-1">Check network connectivity</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <div ref={mapRef} className="h-full w-full" />

      {/* Title overlay */}
      <div className="pointer-events-none absolute top-3 left-3 z-[500] flex items-center gap-2 panel-glass border border-border rounded-md px-2.5 py-1.5">
        <Globe size={11} className="text-muted-foreground" />
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
          Threat Map
        </span>
        <span className="text-[10px] font-mono text-foreground tabular-nums">
          {geoAttacks.length}
        </span>
      </div>

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-[500] panel-glass border border-border rounded-md px-3 py-2 space-y-1">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
          Severity
        </p>
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

export default MapPanel;
