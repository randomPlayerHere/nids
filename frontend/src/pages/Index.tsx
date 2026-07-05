import { useState, useEffect, useCallback, useRef, lazy, Suspense } from 'react';
import NavBar from '@/components/dashboard/NavBar';
import AlertFeed from '@/components/dashboard/AlertFeed';

// Lazy-loaded: pulls three.js / three-globe into a separate chunk so the
// initial bundle stays small.
const GlobePanel = lazy(() => import('@/components/dashboard/GlobePanel'));
import StatsPanel from '@/components/dashboard/StatsPanel';
import BottomBar from '@/components/dashboard/BottomBar';
import ExplanationDrawer from '@/components/dashboard/ExplanationDrawer';
import UploadModal from '@/components/dashboard/UploadModal';
import WelcomeOverlay from '@/components/dashboard/WelcomeOverlay';
import { Alert, generateAlert, enrichGeo } from '@/lib/nids-data';
import { connectAlertStream, StreamStatus } from '@/lib/api';

const Index = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [activeTab, setActiveTab] = useState('Live Demo');
  const [paused, setPaused] = useState(false);
  const [started, setStarted] = useState(false);
  const [status, setStatus] = useState<StreamStatus>('connecting');
  const pausedRef = useRef(paused);

  useEffect(() => { pausedRef.current = paused; }, [paused]);

  const pushAlert = useCallback((alert: Alert) => {
    if (pausedRef.current) return;
    setAlerts(prev => [alert, ...prev].slice(0, 200));
  }, []);

  // Live stream from the backend's /ws/alerts demo feed.
  useEffect(() => {
    if (!started) return;
    const stream = connectAlertStream(pushAlert, setStatus);
    return () => stream.close();
  }, [started, pushAlert]);

  // Manual demo control: inject a synthetic high-severity attack locally.
  const injectAttack = () => {
    const alert = generateAlert();
    const types = ['DDoS', 'Infiltration', 'SSH-Patator', 'Web Attack'];
    const idx = Math.floor(Math.random() * types.length);
    alert.type = types[idx];
    alert.severity = idx < 2 ? 'critical' : 'high';
    alert.confidence = +(0.95 + Math.random() * 0.049).toFixed(3);
    setAlerts(prev => [enrichGeo(alert), ...prev].slice(0, 200));
  };

  // Results from an uploaded CSV replace the feed and pause the live stream.
  const handleAnalyzed = (uploaded: Alert[]) => {
    setPaused(true);
    setAlerts(uploaded.slice(0, 200));
    setActiveTab('Live Demo');
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-background overflow-hidden relative" style={{ minWidth: 1280 }}>
      <NavBar activeTab={activeTab} onTabChange={setActiveTab} />

      <div
        className="flex-1 grid overflow-hidden"
        style={{ gridTemplateColumns: '360px 1fr 340px' }}
      >
        <AlertFeed
          alerts={alerts}
          onSelectAlert={setSelectedAlert}
          selectedId={selectedAlert?.id}
        />
        <div className="relative overflow-hidden">
          <Suspense fallback={<div className="h-full w-full bg-background" />}>
            <GlobePanel alerts={alerts} />
          </Suspense>
          {/* Subtle vignette over globe to tie palette together */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              boxShadow: 'inset 0 0 120px hsl(var(--background))',
            }}
          />
        </div>
        {/* Right column: telemetry panel, with the explanation overlaid on top
            of it (and only it) when an alert is selected. */}
        <div className="relative overflow-hidden">
          <StatsPanel alerts={alerts} />
          <ExplanationDrawer alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
        </div>
      </div>

      <BottomBar
        paused={paused}
        onTogglePause={() => setPaused(p => !p)}
        onInjectAttack={injectAttack}
        totalFlows={alerts.length}
        status={status}
      />

      <UploadModal
        open={activeTab === 'Upload Mode'}
        onClose={() => setActiveTab('Live Demo')}
        onAnalyzed={handleAnalyzed}
      />

      {!started && <WelcomeOverlay onStart={() => setStarted(true)} />}
    </div>
  );
};

export default Index;
