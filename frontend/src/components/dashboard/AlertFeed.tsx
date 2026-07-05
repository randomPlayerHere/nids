import { FC, useMemo, useState } from 'react';
import {
  Zap,
  Flame,
  KeyRound,
  FolderLock,
  Radar,
  Globe,
  Bot,
  CircleCheck,
  ArrowRight,
  Filter,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Alert, SEVERITY_COLORS } from '@/lib/nids-data';

interface AlertFeedProps {
  alerts: Alert[];
  onSelectAlert: (alert: Alert) => void;
  selectedId?: string | null;
}

// Keyed by the model's actual class labels (see models/new/class_labels.json).
const TYPE_ICON: Record<string, LucideIcon> = {
  DDoS: Zap,
  'DoS GoldenEye': Flame,
  'DoS Hulk': Flame,
  'DoS Slowhttptest': Flame,
  'DoS slowloris': Flame,
  'SSH-Patator': KeyRound,
  'FTP-Patator': FolderLock,
  PortScan: Radar,
  'Web Attacks': Globe,
  Botnet: Bot,
  BENIGN: CircleCheck,
};

type FilterMode = 'all' | 'attacks' | 'benign';

const AlertFeed: FC<AlertFeedProps> = ({ alerts, onSelectAlert, selectedId }) => {
  const [filter, setFilter] = useState<FilterMode>('all');

  const filtered = useMemo(() => {
    if (filter === 'attacks') return alerts.filter(a => a.severity !== 'low');
    if (filter === 'benign') return alerts.filter(a => a.severity === 'low');
    return alerts;
  }, [alerts, filter]);

  const counts = useMemo(() => ({
    all: alerts.length,
    attacks: alerts.filter(a => a.severity !== 'low').length,
    benign: alerts.filter(a => a.severity === 'low').length,
  }), [alerts]);

  return (
    <div className="h-full panel-glass border-r border-border flex flex-col overflow-hidden">
      <div className="px-3 pt-3 pb-2 border-b border-border shrink-0 space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter size={11} className="text-muted-foreground" />
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Alert Feed
            </span>
          </div>
          <span className="text-[10px] font-mono text-muted-foreground tabular-nums">
            {filtered.length.toLocaleString()}
          </span>
        </div>

        <div className="flex items-center gap-1 rounded-md border border-border bg-card/40 p-0.5">
          <FilterChip active={filter === 'all'} onClick={() => setFilter('all')} label="All" count={counts.all} />
          <FilterChip
            active={filter === 'attacks'}
            onClick={() => setFilter('attacks')}
            label="Attacks"
            count={counts.attacks}
            tone="danger"
          />
          <FilterChip
            active={filter === 'benign'}
            onClick={() => setFilter('benign')}
            label="Benign"
            count={counts.benign}
            tone="muted"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-2 space-y-1.5">
        {filtered.length === 0 ? (
          <EmptyState filter={filter} />
        ) : (
          filtered.map(alert => (
            <AlertCard
              key={alert.id}
              alert={alert}
              selected={selectedId === alert.id}
              onClick={() => onSelectAlert(alert)}
            />
          ))
        )}
      </div>
    </div>
  );
};

const FilterChip: FC<{
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
  tone?: 'danger' | 'muted' | 'default';
}> = ({ label, count, active, onClick, tone = 'default' }) => {
  const toneColor =
    tone === 'danger' ? 'text-[hsl(var(--severity-critical))]' :
    tone === 'muted' ? 'text-muted-foreground' :
    'text-foreground';
  return (
    <button
      onClick={onClick}
      className={`focus-ring flex-1 flex items-center justify-center gap-1.5 px-2 py-1 rounded text-[11px] font-medium transition-all duration-150 active:scale-[0.97] ${
        active ? 'bg-card-elevated text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
      }`}
    >
      {label}
      <span className={`font-mono text-[10px] tabular-nums ${active ? toneColor : 'text-muted-foreground/70'}`}>
        {count}
      </span>
    </button>
  );
};

const EmptyState: FC<{ filter: FilterMode }> = ({ filter }) => (
  <div className="flex flex-col items-center justify-center h-full text-center px-4">
    <div className="h-9 w-9 rounded-full border border-border bg-card/50 flex items-center justify-center mb-3">
      <Radar size={15} className="text-muted-foreground animate-ticker" />
    </div>
    <p className="text-[12px] text-foreground font-medium">Listening for flows…</p>
    <p className="text-[10px] text-muted-foreground mt-1">
      {filter === 'attacks' ? 'No attacks yet — that\'s a good thing.' :
       filter === 'benign' ? 'Benign traffic will appear here.' :
       'Incoming alerts will stream in real time.'}
    </p>
  </div>
);

const AlertCard: FC<{ alert: Alert; onClick: () => void; selected?: boolean }> = ({ alert, onClick, selected }) => {
  const color = SEVERITY_COLORS[alert.severity];
  const isAttack = alert.severity !== 'low';
  const Icon = TYPE_ICON[alert.type] || CircleCheck;
  const conf = (alert.confidence * 100).toFixed(1);

  return (
    <button
      onClick={onClick}
      className={`group focus-ring w-full text-left rounded-md px-2.5 py-2 transition-all duration-150 animate-slide-in-left
        ${selected
          ? 'bg-card-elevated ring-1 ring-foreground/20'
          : 'bg-card hover:bg-card-elevated'}
        ${isAttack ? 'animate-glow-pulse' : ''}`}
      style={{
        boxShadow: `inset 2px 0 0 0 ${color}`,
      }}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <Icon size={11} style={{ color }} strokeWidth={2.25} />
          <span
            className="text-[10.5px] font-semibold truncate"
            style={{ color }}
          >
            {alert.type}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-[10px] font-mono text-muted-foreground tabular-nums">{conf}%</span>
          <ArrowRight size={11} className="text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="text-[10px] font-mono text-muted-foreground truncate min-w-0">
          {alert.srcIP} <span className="text-muted-foreground/40">→</span> {alert.dstIP}
        </div>
        <div className="text-[9.5px] font-mono text-muted-foreground/60 shrink-0">
          {alert.timestamp.toLocaleTimeString('en-US', { hour12: false })}
        </div>
      </div>
    </button>
  );
};

export default AlertFeed;
