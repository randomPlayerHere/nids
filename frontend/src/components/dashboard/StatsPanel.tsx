import { FC, useMemo, useEffect, useRef, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip as ChartTooltip,
} from 'chart.js';
import { TrendingUp, TrendingDown, Minus, Activity, AlertTriangle, Target, Flame } from 'lucide-react';
import { Alert, SEVERITY_COLORS } from '@/lib/nids-data';

ChartJS.register(CategoryScale, LinearScale, BarElement, ChartTooltip);

interface StatsPanelProps {
  alerts: Alert[];
}

function severityColor(type: string, severity: Alert['severity']) {
  return SEVERITY_COLORS[severity];
}

const StatsPanel: FC<StatsPanelProps> = ({ alerts }) => {
  const prevTotal = useRef(0);
  const [trend, setTrend] = useState<'up' | 'down' | 'flat'>('flat');

  useEffect(() => {
    if (alerts.length > prevTotal.current) setTrend('up');
    else if (alerts.length < prevTotal.current) setTrend('down');
    prevTotal.current = alerts.length;
  }, [alerts.length]);

  const stats = useMemo(() => {
    const attacks = alerts.filter(a => a.severity !== 'low');
    const typeCounts: Record<string, number> = {};
    attacks.forEach(a => { typeCounts[a.type] = (typeCounts[a.type] || 0) + 1; });
    const sortedTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
    const topType = sortedTypes[0];

    // Full 0-100% range in 20% bins. 11-class softmax often yields sub-50%
    // confidence, so the old 50-100% buckets silently dropped many flows.
    const confBuckets = [0, 0, 0, 0, 0];
    alerts.forEach(a => {
      const pct = a.confidence * 100;
      const idx = Math.min(Math.floor(pct / 20), 4);
      if (idx >= 0) confBuckets[idx]++;
    });

    // Sparkline data: bucket last 30 alerts into 10 bins
    const recent = alerts.slice(0, 30).reverse();
    const sparkBins = Array.from({ length: 10 }, (_, i) => {
      const start = Math.floor((i / 10) * recent.length);
      const end = Math.floor(((i + 1) / 10) * recent.length);
      return recent.slice(start, end).filter(a => a.severity !== 'low').length;
    });

    const avgConf = alerts.length
      ? (alerts.reduce((s, a) => s + a.confidence, 0) / alerts.length) * 100
      : 0;

    return {
      total: alerts.length,
      attacks: attacks.length,
      rate: alerts.length ? (attacks.length / alerts.length) * 100 : 0,
      topType: topType ? topType[0] : '—',
      topTypeCount: topType ? topType[1] : 0,
      typeCounts,
      sortedTypes,
      confBuckets,
      sparkBins,
      avgConf,
    };
  }, [alerts]);

  return (
    <div className="h-full panel-glass border-l border-border flex flex-col overflow-hidden">
      <div className="px-3 py-3 border-b border-border shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={11} className="text-muted-foreground" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Telemetry
          </span>
        </div>
        <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-success animate-dot-pulse" />
          Live
        </span>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-3">
        {/* Hero metric */}
        <div className="rounded-lg border border-border bg-card/60 p-3.5">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                Flows analyzed
              </p>
              <p className="font-mono text-[28px] font-bold leading-none mt-1.5 tabular-nums tracking-tight">
                {stats.total.toLocaleString()}
              </p>
            </div>
            <TrendBadge trend={trend} />
          </div>
          <Sparkline data={stats.sparkBins} />
        </div>

        {/* 2x2 metric grid */}
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            icon={AlertTriangle}
            label="Attacks"
            value={stats.attacks.toLocaleString()}
            tone="danger"
          />
          <MetricCard
            icon={Target}
            label="Attack rate"
            value={`${stats.rate.toFixed(1)}%`}
            tone={stats.rate > 30 ? 'danger' : stats.rate > 10 ? 'warn' : 'default'}
          />
          <MetricCard
            icon={Flame}
            label="Top threat"
            value={stats.topType}
            sub={stats.topTypeCount ? `${stats.topTypeCount} flows` : ''}
            small
          />
          <MetricCard
            icon={Activity}
            label="Avg conf."
            value={`${stats.avgConf.toFixed(1)}%`}
          />
        </div>

        {/* Attack breakdown */}
        <Section title="Attack breakdown">
          {stats.sortedTypes.length > 0 ? (
            <div className="space-y-1.5">
              {stats.sortedTypes.slice(0, 6).map(([type, count]) => {
                const total = stats.attacks || 1;
                const pct = (count / total) * 100;
                const color = severityColor(type, getSeverityForType(type));
                return (
                  <div key={type} className="space-y-1">
                    <div className="flex items-center justify-between text-[10.5px]">
                      <span className="font-medium text-foreground truncate pr-2">{type}</span>
                      <span className="font-mono text-muted-foreground tabular-nums shrink-0">
                        {count} <span className="text-muted-foreground/50">· {pct.toFixed(0)}%</span>
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-secondary/60 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, background: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-[11px] text-muted-foreground py-4 text-center">No attacks detected yet</p>
          )}
        </Section>

        {/* Confidence histogram */}
        <Section title="Confidence distribution">
          <div className="h-[96px]">
            <Bar
              data={{
                labels: ['0-20', '20-40', '40-60', '60-80', '80-100'],
                datasets: [{
                  data: stats.confBuckets,
                  backgroundColor: 'hsl(0 0% 60% / 0.18)',
                  borderColor: 'hsl(0 0% 70%)',
                  borderWidth: 1,
                  borderRadius: 3,
                }],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: {
                    grid: { display: false },
                    ticks: { color: 'hsl(220 10% 55%)', font: { family: 'JetBrains Mono', size: 9 } },
                  },
                  y: {
                    grid: { color: 'hsl(222 14% 17% / 0.6)' },
                    ticks: { color: 'hsl(220 10% 55%)', font: { family: 'JetBrains Mono', size: 9 }, stepSize: 1 },
                  },
                },
              }}
            />
          </div>
        </Section>

        {/* Model card */}
        <div className="rounded-lg border border-border/60 bg-card/30 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Model</span>
            <span className="chip">DCNN · 1D</span>
          </div>
          <div className="space-y-1 text-[10.5px] font-mono text-muted-foreground">
            <KV k="Features" v="78" />
            <KV k="Classes" v="11" />
            <KV k="Trained on" v="CICIDS2017" />
            <KV k="Latency p50" v="3.2 ms" />
          </div>
        </div>
      </div>
    </div>
  );
};

const Section: FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="rounded-lg border border-border bg-card/40 p-3">
    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2.5">{title}</p>
    {children}
  </div>
);

const KV: FC<{ k: string; v: string }> = ({ k, v }) => (
  <div className="flex items-center justify-between">
    <span>{k}</span>
    <span className="text-foreground">{v}</span>
  </div>
);

const TrendBadge: FC<{ trend: 'up' | 'down' | 'flat' }> = ({ trend }) => {
  const map = {
    up: { icon: TrendingUp, color: 'hsl(var(--success))', bg: 'hsl(var(--success) / 0.1)' },
    down: { icon: TrendingDown, color: 'hsl(var(--muted-foreground))', bg: 'hsl(var(--muted) / 0.4)' },
    flat: { icon: Minus, color: 'hsl(var(--muted-foreground))', bg: 'hsl(var(--muted) / 0.4)' },
  }[trend];
  const Icon = map.icon;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-mono"
      style={{ color: map.color, background: map.bg }}
    >
      <Icon size={10} />
      live
    </span>
  );
};

const Sparkline: FC<{ data: number[] }> = ({ data }) => {
  const max = Math.max(...data, 1);
  return (
    <div className="mt-3 flex items-end gap-[3px] h-7">
      {data.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm bg-muted-foreground/40 transition-all duration-300"
          style={{ height: `${(v / max) * 100}%`, minHeight: 2 }}
        />
      ))}
    </div>
  );
};

const MetricCard: FC<{
  icon: typeof Activity;
  label: string;
  value: string;
  sub?: string;
  tone?: 'default' | 'danger' | 'warn';
  small?: boolean;
}> = ({ icon: Icon, label, value, sub, tone = 'default', small }) => {
  const accent =
    tone === 'danger' ? 'text-[hsl(var(--severity-critical))]' :
    tone === 'warn' ? 'text-[hsl(var(--severity-high))]' :
    'text-muted-foreground';
  return (
    <div className="rounded-lg border border-border bg-card/40 p-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[9.5px] font-medium text-muted-foreground uppercase tracking-wider truncate pr-1">
          {label}
        </span>
        <Icon size={11} className={accent} />
      </div>
      <p className={`font-mono font-bold text-foreground tabular-nums leading-none ${small ? 'text-[13px]' : 'text-[18px]'}`}>
        {value}
      </p>
      {sub && <p className="text-[9.5px] text-muted-foreground mt-1 font-mono">{sub}</p>}
    </div>
  );
};

function getSeverityForType(type: string): Alert['severity'] {
  const map: Record<string, Alert['severity']> = {
    DDoS: 'critical',
    Infiltration: 'critical',
    'SSH-Patator': 'high',
    'FTP-Patator': 'high',
    PortScan: 'medium',
    'Web Attack': 'medium',
    Botnet: 'medium',
  };
  return map[type] || 'low';
}

export default StatsPanel;
