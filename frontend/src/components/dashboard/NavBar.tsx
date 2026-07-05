import { FC } from 'react';
import { Activity, Upload, BookOpen, Github } from 'lucide-react';
import Logo from './Logo';

interface NavBarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: 'Live Demo', icon: Activity, label: 'Live' },
  { id: 'Upload Mode', icon: Upload, label: 'Upload' },
  { id: 'API Docs', icon: BookOpen, label: 'API' },
] as const;

const NavBar: FC<NavBarProps> = ({ activeTab, onTabChange }) => {
  const handleTabClick = (tab: string) => {
    if (tab === 'API Docs') {
      window.open('/docs', '_blank');
      return;
    }
    onTabChange(tab);
  };

  return (
    <nav className="h-[56px] panel-glass border-b border-border flex items-center justify-between px-4 shrink-0 relative z-30">
      <div className="flex items-center gap-7">
        <div className="flex items-center gap-2.5">
          <Logo size={22} />
          <div className="flex items-baseline gap-2">
            <span className="font-semibold text-[15px] tracking-tight text-foreground">Sentinel</span>
            <span className="text-[11px] font-mono text-muted-foreground hidden sm:inline">NIDS</span>
            <span className="chip ml-1 hidden md:inline-flex">v0.2 · beta</span>
          </div>
        </div>

        <div className="flex items-center gap-0.5 rounded-lg border border-border bg-card/40 p-0.5">
          {TABS.map(({ id, icon: Icon, label }) => {
            const active = id !== 'API Docs' && activeTab === id;
            return (
              <button
                key={id}
                onClick={() => handleTabClick(id)}
                className={`focus-ring flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] font-medium rounded-md transition-all duration-150 active:scale-[0.97] ${
                  active
                    ? 'bg-card-elevated text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground hover:bg-card-elevated/60'
                }`}
              >
                <Icon size={13} strokeWidth={2} />
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <a
          href="https://github.com/randomPlayerHere/nids"
          target="_blank"
          rel="noreferrer"
          className="focus-ring text-muted-foreground hover:text-foreground transition-colors p-1.5 rounded-md hover:bg-card-elevated/60"
          aria-label="GitHub repository"
        >
          <Github size={15} />
        </a>

        <div className="h-5 w-px bg-border" />

        <div className="flex items-center gap-2 rounded-full border border-success/30 bg-success/5 px-2.5 py-1">
          <span className="relative inline-flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 animate-pulse-ring" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success" />
          </span>
          <span className="text-[11px] font-mono text-success">Online</span>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
