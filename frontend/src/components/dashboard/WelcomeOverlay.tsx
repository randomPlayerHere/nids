import { FC } from 'react';
import { Play, ShieldCheck, Globe2, BarChart3, Sparkles, ArrowRight, Github } from 'lucide-react';
import Logo from './Logo';

interface WelcomeOverlayProps {
  onStart: () => void;
}

const FEATURES = [
  { icon: ShieldCheck, label: 'Real-time threat detection' },
  { icon: Globe2, label: 'Geographic threat mapping' },
  { icon: BarChart3, label: 'Per-flow telemetry' },
  { icon: Sparkles, label: 'Explainable AI (SHAP)' },
];

const WelcomeOverlay: FC<WelcomeOverlayProps> = ({ onStart }) => {
  return (
    <div
      className="fixed inset-x-0 bottom-0 z-[10000] flex items-center justify-center p-6"
      style={{ top: 56, background: 'hsl(var(--background) / 0.7)', backdropFilter: 'blur(12px)' }}
    >
      {/* Background grid */}
      <div className="absolute inset-0 grid-bg opacity-20 pointer-events-none" />

      <div className="relative w-full max-w-[560px] rounded-2xl border border-border panel-glass p-8 text-center shadow-2xl animate-fade-in-up">
        <div className="mx-auto mb-5 inline-flex items-center justify-center h-14 w-14 rounded-2xl border border-border bg-card/80 shadow-inner">
          <Logo size={28} />
        </div>

        <div className="flex items-center justify-center gap-2 mb-3">
          <span className="chip">
            <span className="h-1.5 w-1.5 rounded-full bg-success animate-dot-pulse" />
            Demo environment
          </span>
          <span className="chip">v0.2 · beta</span>
        </div>

        <h1 className="text-[26px] font-bold leading-tight tracking-tight text-foreground">
          Detect network intrusions in real time
        </h1>
        <p className="text-[14px] text-muted-foreground mt-3 leading-relaxed max-w-[420px] mx-auto">
          A 1D deep-CNN trained on CICIDS2017 classifies flows across 11 threat categories — with
          per-prediction feature attributions.
        </p>

        <div className="grid grid-cols-2 gap-2 mt-6 mb-7">
          {FEATURES.map(({ icon: Icon, label }) => (
            <div
              key={label}
              className="flex items-center gap-2 rounded-lg border border-border bg-card/40 px-3 py-2 text-left"
            >
              <Icon size={13} className="text-muted-foreground shrink-0" />
              <span className="text-[11.5px] text-foreground/90">{label}</span>
            </div>
          ))}
        </div>

        <button
          onClick={onStart}
          className="focus-ring group w-full inline-flex items-center justify-center gap-2 rounded-lg py-3 text-[13.5px] font-semibold text-primary-foreground bg-primary hover:brightness-95 transition-all active:scale-[0.99]"
        >
          <Play size={13} fill="currentColor" />
          Start live demo
          <ArrowRight size={13} className="group-hover:translate-x-0.5 transition-transform" />
        </button>

        <div className="mt-4 flex items-center justify-center gap-4 text-[11px] text-muted-foreground">
          <a
            href="https://github.com/randomPlayerHere/nids"
            target="_blank"
            rel="noreferrer"
            className="focus-ring inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
          >
            <Github size={11} />
            View on GitHub
          </a>
          <span className="text-muted-foreground/40">·</span>
          <span>Simulated data — no real traffic is analyzed</span>
        </div>
      </div>
    </div>
  );
};

export default WelcomeOverlay;
