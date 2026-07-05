import { FC, useState, useEffect, useRef } from 'react';
import { Pause, Play, Zap, Cpu, Clock, Wifi } from 'lucide-react';
import { toast } from 'sonner';
import type { StreamStatus } from '@/lib/api';

interface BottomBarProps {
  paused: boolean;
  onTogglePause: () => void;
  onInjectAttack: () => void;
  totalFlows: number;
  status?: StreamStatus;
}

const STATUS_META: Record<StreamStatus, { label: string; color: string }> = {
  open: { label: 'Live', color: 'hsl(var(--success))' },
  connecting: { label: 'Connecting', color: 'hsl(var(--severity-high))' },
  closed: { label: 'Offline', color: 'hsl(var(--severity-critical))' },
};

const BottomBar: FC<BottomBarProps> = ({ paused, onTogglePause, onInjectAttack, totalFlows, status = 'connecting' }) => {
  const [flashing, setFlashing] = useState(false);
  const [throughput, setThroughput] = useState(0);
  const lastTotal = useRef(totalFlows);
  const lastTime = useRef(Date.now());

  useEffect(() => {
    const t = setInterval(() => {
      const now = Date.now();
      const dt = (now - lastTime.current) / 1000;
      const dN = totalFlows - lastTotal.current;
      const rate = dt > 0 ? dN / dt : 0;
      setThroughput(prev => prev * 0.6 + rate * 0.4);
      lastTotal.current = totalFlows;
      lastTime.current = now;
    }, 1000);
    return () => clearInterval(t);
  }, [totalFlows]);

  const handleInject = () => {
    setFlashing(true);
    onInjectAttack();
    toast.success('Synthetic attack injected', {
      description: 'A high-confidence threat has been added to the feed.',
    });
    setTimeout(() => setFlashing(false), 600);
  };

  return (
    <div className="h-[44px] panel-glass border-t border-border flex items-center justify-between px-3 shrink-0 text-[11px]">
      <div className="flex items-center gap-2">
        <button
          onClick={onTogglePause}
          className={`focus-ring inline-flex items-center gap-1.5 px-3 py-1.5 text-[11.5px] font-semibold rounded-md transition-colors active:scale-[0.97] ${
            paused
              ? 'bg-primary text-primary-foreground hover:brightness-95'
              : 'bg-card border border-border text-foreground hover:bg-card-elevated'
          }`}
        >
          {paused ? <Play size={11} fill="currentColor" /> : <Pause size={11} fill="currentColor" />}
          {paused ? 'Resume stream' : 'Pause'}
        </button>
        <button
          onClick={handleInject}
          className={`focus-ring inline-flex items-center gap-1.5 px-3 py-1.5 text-[11.5px] font-medium rounded-md border transition-colors active:scale-[0.97] ${
            flashing
              ? 'animate-flash-accent border-foreground/40 text-foreground'
              : 'border-border bg-card hover:bg-card-elevated text-foreground'
          }`}
        >
          <Zap size={11} className="text-[hsl(var(--severity-high))]" />
          Inject attack
        </button>
      </div>

      <div className="flex items-center gap-4 text-muted-foreground">
        <div className="hidden sm:flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: STATUS_META[status].color }} />
          <span className="font-mono text-foreground tabular-nums">{STATUS_META[status].label}</span>
        </div>
        <div className="h-3.5 w-px bg-border" />
        <Stat icon={Wifi} label="Throughput" value={`${throughput.toFixed(1)} flows/s`} accent={throughput > 0.5} />
        <div className="h-3.5 w-px bg-border" />
        <Stat icon={Cpu} label="Model" value="nids_dcnn.v0.2" />
        <div className="h-3.5 w-px bg-border" />
        <Stat icon={Clock} label="Latency" value="3.2 ms p50" />
      </div>
    </div>
  );
};

const Stat: FC<{ icon: typeof Cpu; label: string; value: string; accent?: boolean }> = ({
  icon: Icon,
  label,
  value,
  accent,
}) => (
  <div className="hidden sm:flex items-center gap-1.5">
    <Icon size={11} className={accent ? 'text-success' : 'text-muted-foreground/70'} />
    <span className="text-muted-foreground/70">{label}</span>
    <span className="font-mono text-foreground tabular-nums">{value}</span>
  </div>
);

export default BottomBar;
