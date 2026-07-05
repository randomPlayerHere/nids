import { FC, useState, useRef, useEffect } from 'react';
import { UploadCloud, FileText, X, Sparkles, FileWarning, CheckCircle2, ArrowRight } from 'lucide-react';
import { analyzeCsv, AnalyzeSummary } from '@/lib/api';
import { Alert } from '@/lib/nids-data';

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onAnalyzed: (alerts: Alert[]) => void;
}

const ACCEPTED = ['.csv'];

const UploadModal: FC<UploadModalProps> = ({ open, onClose, onAnalyzed }) => {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [includeShap, setIncludeShap] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<AnalyzeSummary | null>(null);
  const [results, setResults] = useState<Alert[] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => e.key === 'Escape' && handleClose();
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const handleClose = () => {
    setFile(null);
    setAnalyzing(false);
    setDone(false);
    setError(null);
    setSummary(null);
    setResults(null);
    onClose();
  };

  const acceptFile = (f: File) => {
    const ok = ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext));
    if (!ok) {
      setError(`Unsupported file. Use ${ACCEPTED.join(', ')}.`);
      return;
    }
    setError(null);
    setFile(f);
    setDone(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) acceptFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    try {
      const { alerts, summary } = await analyzeCsv(file, includeShap);
      setResults(alerts);
      setSummary(summary);
      setDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed. Is the backend running?');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleViewResults = () => {
    if (results) onAnalyzed(results);
    handleClose();
  };

  const fileSize = file ? (file.size / 1024).toFixed(1) + ' KB' : '';

  return (
    <div
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{ background: 'hsl(var(--background) / 0.75)', backdropFilter: 'blur(8px)', zIndex: 1000 }}
      onClick={handleClose}
    >
      <div
        className="relative w-full max-w-[520px] rounded-xl border border-border panel-glass shadow-2xl animate-fade-in-up"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 px-5 pt-5 pb-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h2 className="text-[15px] font-semibold text-foreground">Analyze network traffic</h2>
              <span className="chip">Beta</span>
            </div>
            <p className="text-[12px] text-muted-foreground leading-relaxed">
              Drop a CICIDS-format CSV. Sentinel will run inference and surface flagged flows with explanations.
            </p>
          </div>
          <button
            onClick={handleClose}
            className="focus-ring shrink-0 p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-card-elevated/60 transition-colors active:scale-[0.95]"
            aria-label="Close"
          >
            <X size={15} />
          </button>
        </div>

        <div className="px-5 pb-5 space-y-4">
          {/* Drop zone */}
          {!file ? (
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              className={`relative cursor-pointer rounded-lg border-2 border-dashed transition-all duration-200 grid-bg overflow-hidden
                ${dragging
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-card/30 hover:border-muted-foreground/60 hover:bg-card/50'}`}
            >
              <div className="px-6 py-10 flex flex-col items-center text-center relative">
                <div className={`h-12 w-12 rounded-full border flex items-center justify-center mb-3 transition-all ${
                  dragging ? 'border-primary bg-primary/10 scale-110' : 'border-border bg-card'
                }`}>
                  <UploadCloud size={20} className={dragging ? 'text-primary' : 'text-muted-foreground'} />
                </div>
                <p className="text-[13px] font-medium text-foreground">
                  {dragging ? 'Drop to upload' : 'Drop file or click to browse'}
                </p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  CICIDS-format CSV ({ACCEPTED.join(' · ')})
                </p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED.join(',')}
                className="hidden"
                onChange={e => e.target.files?.[0] && acceptFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="rounded-lg border border-border bg-card/60 p-3.5 flex items-center gap-3">
              <div className="h-10 w-10 rounded-md border border-border bg-card flex items-center justify-center shrink-0">
                <FileText size={15} className="text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium text-foreground truncate">{file.name}</p>
                <p className="text-[11px] font-mono text-muted-foreground">{fileSize}</p>
              </div>
              {!analyzing && !done && (
                <button
                  onClick={() => setFile(null)}
                  className="focus-ring shrink-0 p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-card-elevated transition-colors"
                  aria-label="Remove file"
                >
                  <X size={14} />
                </button>
              )}
              {done && <CheckCircle2 size={16} className="text-success shrink-0" />}
            </div>
          )}

          {/* Options */}
          <div className="rounded-lg border border-border bg-card/30 p-3">
            <label className="flex items-center justify-between gap-3 cursor-pointer group">
              <div className="flex items-start gap-2.5">
                <Sparkles size={13} className="text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="text-[12px] font-medium text-foreground">SHAP explanations</p>
                  <p className="text-[10.5px] text-muted-foreground leading-relaxed">
                    Compute per-feature attributions for each flagged flow. Adds ~2s to analysis.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIncludeShap(!includeShap)}
                className={`focus-ring relative shrink-0 h-5 w-9 rounded-full transition-colors ${
                  includeShap ? 'bg-primary' : 'bg-secondary'
                }`}
                aria-pressed={includeShap}
              >
                <span
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-foreground shadow-sm transition-all ${
                    includeShap ? 'left-[18px]' : 'left-0.5'
                  }`}
                />
              </button>
            </label>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-[11.5px] text-destructive">
              <FileWarning size={13} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Done summary */}
          {done && summary && (
            <div className="flex items-start gap-2 rounded-md border border-success/30 bg-success/5 px-3 py-2.5 text-[11.5px]">
              <CheckCircle2 size={14} className="text-success mt-0.5 shrink-0" />
              <div className="text-foreground/90 flex-1">
                <p className="font-medium">Analysis complete</p>
                <p className="text-muted-foreground text-[10.5px] mt-0.5">
                  {summary.total} flows · <span className="text-success">{summary.benign} benign</span> ·{' '}
                  <span className="text-destructive">{summary.malicious} malicious</span>
                </p>
                {summary.malicious > 0 && (
                  <p className="text-muted-foreground text-[10px] mt-1 font-mono">
                    {Object.entries(summary.by_class)
                      .filter(([k]) => k !== 'BENIGN')
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 4)
                      .map(([k, v]) => `${k}:${v}`)
                      .join('  ')}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Action */}
          <button
            onClick={done ? handleViewResults : handleAnalyze}
            disabled={!file || analyzing}
            className={`focus-ring w-full inline-flex items-center justify-center gap-2 rounded-md py-2.5 text-[12.5px] font-semibold transition-all duration-150 active:scale-[0.99] ${
              !file || analyzing
                ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                : done
                ? 'bg-success text-background hover:brightness-110'
                : 'bg-primary text-primary-foreground hover:brightness-95'
            }`}
          >
            {analyzing ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing…
              </>
            ) : done ? (
              <>View results <ArrowRight size={13} /></>
            ) : (
              <>Run analysis <ArrowRight size={13} /></>
            )}
          </button>

          <p className="text-[10px] text-center text-muted-foreground/60">
            CSV is sent to the NIDS backend for inference. Results replace the live feed.
          </p>
        </div>
      </div>
    </div>
  );
};

export default UploadModal;
