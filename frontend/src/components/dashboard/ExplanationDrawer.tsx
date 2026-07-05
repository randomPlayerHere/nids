import { FC, useEffect } from 'react';
import { X, ShieldAlert, Copy, MapPin, Network, Clock, Hash, ArrowRight, Info } from 'lucide-react';
import { toast } from 'sonner';
import { Alert, SEVERITY_COLORS, getSeverityLabel } from '@/lib/nids-data';

interface ExplanationDrawerProps {
  alert: Alert | null;
  onClose: () => void;
}

const SEVERITY_ACTIONS: Record<string, string> = {
  critical: 'Block source IP and alert your network team immediately.',
  high: 'Investigate source IP and review authentication logs.',
  medium: 'Log and monitor. Consider rate limiting this source.',
  low: 'No action required. Routine traffic.',
};

const ExplanationDrawer: FC<ExplanationDrawerProps> = ({ alert, onClose }) => {
  // Close on escape
  useEffect(() => {
    if (!alert) return;
    const handler = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [alert, onClose]);

  if (!alert) return null;

  const color = SEVERITY_COLORS[alert.severity];
  // Only show real SHAP attributions computed by the backend (Upload Mode).
  // Live-stream alerts carry none, so we show a note rather than fabricated values.
  const shap = alert.shapValues && alert.shapValues.length > 0
    ? [...alert.shapValues].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 6)
    : [];
  const maxAbs = Math.max(...shap.map(s => Math.abs(s.value)), 0.01);

  const copyJSON = () => {
    const payload = {
      id: alert.id,
      type: alert.type,
      severity: alert.severity,
      confidence: alert.confidence,
      timestamp: alert.timestamp.toISOString(),
      flow: {
        src: alert.srcIP,
        dst: alert.dstIP,
        protocol: alert.protocol,
        duration_us: alert.flowDuration,
        fwd_packets: alert.fwdPackets,
      },
      geo: alert.geo,
    };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    toast.success('Alert JSON copied to clipboard');
  };

  return (
    <div className="absolute inset-0 z-30 flex flex-col border-l border-border bg-[hsl(var(--panel))]">
      {/* Header */}
        <div className="flex items-center justify-between px-4 py-3.5 border-b border-border shrink-0">
          <div className="flex items-center gap-2.5">
            <div
              className="h-8 w-8 rounded-md flex items-center justify-center border"
              style={{ background: `${color}18`, borderColor: `${color}40` }}
            >
              <ShieldAlert size={15} style={{ color }} />
            </div>
            <div>
              <p className="text-[13px] font-semibold text-foreground leading-tight">{alert.type}</p>
              <p className="text-[10.5px] text-muted-foreground font-mono">Alert · {alert.id.slice(-8)}</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={copyJSON}
              className="focus-ring p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-card-elevated/60 transition-colors active:scale-[0.95]"
              title="Copy JSON"
            >
              <Copy size={14} />
            </button>
            <button
              onClick={onClose}
              className="focus-ring p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-card-elevated/60 transition-colors active:scale-[0.95]"
              title="Close (Esc)"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto overflow-x-hidden scrollbar-thin p-4 space-y-4">
          {/* Confidence + severity hero */}
          <div className="rounded-lg border border-border bg-card/60 p-3.5">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <span
                  className="text-[10.5px] font-semibold px-2 py-0.5 rounded uppercase tracking-wider"
                  style={{ background: `${color}22`, color }}
                >
                  {getSeverityLabel(alert.severity)}
                </span>
                <span className="text-[10.5px] font-mono text-muted-foreground">
                  {alert.timestamp.toLocaleString('en-US', { hour12: false })}
                </span>
              </div>
            </div>
            <div className="flex items-end gap-2 mb-2">
              <p className="font-mono text-[28px] font-bold leading-none tabular-nums" style={{ color }}>
                {(alert.confidence * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-muted-foreground pb-1">model confidence</p>
            </div>
            <div className="h-1.5 rounded-full bg-secondary/60 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${alert.confidence * 100}%`, background: color }}
              />
            </div>
          </div>

          {/* Flow details */}
          <Section title="Flow details" icon={Network}>
            <dl className="grid grid-cols-2 gap-2.5 text-[11px]">
              <KV icon={ArrowRight} k="Source" v={alert.srcIP} mono />
              <KV icon={ArrowRight} k="Destination" v={alert.dstIP} mono />
              <KV icon={Hash} k="Protocol" v={alert.protocol} />
              <KV icon={Clock} k="Duration" v={`${alert.flowDuration.toLocaleString()} µs`} />
              <KV icon={Hash} k="Fwd packets" v={alert.fwdPackets.toString()} />
              <KV icon={MapPin} k="Location" v={alert.geo?.city || '—'} />
            </dl>
          </Section>

          {/* SHAP — only when the backend actually computed it (Upload Mode) */}
          {shap.length > 0 ? (
            <Section
              title="Feature importance"
              icon={Info}
              right={<span className="chip">SHAP</span>}
            >
              <div className="space-y-2">
                {shap.map(s => {
                  const pct = (Math.abs(s.value) / maxAbs) * 100;
                  const positive = s.value > 0;
                  const barColor = positive ? 'hsl(0 65% 58%)' : 'hsl(0 0% 60%)';
                  const label = positive ? `+${s.value.toFixed(2)}` : s.value.toFixed(2);
                  return (
                    <div key={s.feature} className="space-y-1">
                      <div className="flex items-center justify-between gap-2 text-[10.5px] min-w-0">
                        <span className="font-mono text-foreground/90 truncate min-w-0">{s.feature}</span>
                        <span className="font-mono tabular-nums shrink-0" style={{ color: barColor }}>
                          {label}
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-secondary/40 overflow-hidden relative">
                        <div
                          className="absolute inset-y-0 rounded-full transition-all duration-500"
                          style={{
                            left: positive ? '50%' : `${50 - pct / 2}%`,
                            width: `${pct / 2}%`,
                            background: barColor,
                          }}
                        />
                        <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 pt-3 border-t border-border/60 space-y-1.5">
                {shap.slice(0, 3).map(s => {
                  const desc = (s as { description?: string }).description;
                  return desc ? (
                    <p key={s.feature} className="text-[10.5px] text-muted-foreground leading-relaxed">
                      <span className="font-mono font-medium text-foreground">{s.feature}</span>
                      <span className="text-muted-foreground/50"> · </span>
                      {desc}
                    </p>
                  ) : null;
                })}
              </div>
            </Section>
          ) : (
            <Section title="Feature importance" icon={Info}>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                No SHAP attributions for this flow. They're computed in{' '}
                <span className="text-foreground font-medium">Upload Mode</span> for the first
                flows of an analyzed CSV — open one of those to see why it was flagged.
              </p>
            </Section>
          )}

          {/* Recommended action */}
          <Section title="Recommended action" icon={ShieldAlert}>
            <p className="text-[11.5px] text-foreground/85 leading-relaxed">
              {SEVERITY_ACTIONS[alert.severity]}
            </p>
          </Section>

          <p className="text-[10px] text-muted-foreground/60 text-center pt-2">
            Model prediction is real. Source IP, protocol &amp; location are synthesized for display.
          </p>
        </div>
      </div>
  );
};

const Section: FC<{
  title: string;
  icon: typeof Network;
  right?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, icon: Icon, right, children }) => (
  <div className="rounded-lg border border-border bg-card/40 p-3.5">
    <div className="flex items-center justify-between mb-2.5">
      <div className="flex items-center gap-2">
        <Icon size={11} className="text-muted-foreground" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </span>
      </div>
      {right}
    </div>
    {children}
  </div>
);

const KV: FC<{ icon: typeof Network; k: string; v: string; mono?: boolean }> = ({
  icon: Icon,
  k,
  v,
  mono,
}) => (
  <div className="min-w-0">
    <div className="flex items-center gap-1 text-[9.5px] uppercase tracking-wider text-muted-foreground mb-0.5">
      <Icon size={9} />
      {k}
    </div>
    <p className={`text-foreground truncate ${mono ? 'font-mono' : ''}`}>{v}</p>
  </div>
);

export default ExplanationDrawer;
