import { useEffect } from 'react';
import { pingKeepAlive } from '@/lib/api';

/**
 * Periodically pings the backend so a free-tier Hugging Face Space doesn't go
 * idle while a user has the dashboard open.
 *
 * Spaces sleep after a fixed window of inactivity, so we ping on mount and then
 * on a fixed interval. To avoid hammering the Space while it's hidden (and to
 * play nice with browsers that throttle background timers), we pause pinging
 * when the tab isn't visible and fire one immediately when it comes back.
 *
 * @param intervalMs how often to ping while the tab is visible (default 4 min).
 */
export function useKeepAlive(intervalMs = 4 * 60 * 1000): void {
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;

    const start = () => {
      if (timer !== undefined) return;
      void pingKeepAlive(); // ping right away, then on the interval
      timer = setInterval(() => void pingKeepAlive(), intervalMs);
    };

    const stop = () => {
      if (timer !== undefined) {
        clearInterval(timer);
        timer = undefined;
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === 'visible') start();
      else stop();
    };

    if (document.visibilityState === 'visible') start();
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      stop();
    };
  }, [intervalMs]);
}
