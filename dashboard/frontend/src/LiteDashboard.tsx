import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import type { Signal } from './store'
import { SignalCard } from './components/SignalCard'
import { FeedbackButton } from './components/FeedbackButton'
import { captureLiteEvent } from './analytics'
import './LiteDashboard.css'

type LitePayload = {
  generated_at?: string
  count?: number
  run_id?: string
  signals?: Signal[]
}

const formatTime = (value?: string) => {
  if (!value) return '未知'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN')
}

export const LiteDashboard = () => {
  const UPDATE_INTERVAL_HOURS = 1
  const navigate = useNavigate()
  const location = useLocation()
  const [payload, setPayload] = useState<LitePayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const signalCountRef = useRef(0)

  useEffect(() => {
    signalCountRef.current = payload?.signals?.length ?? 0
  }, [payload])

  useEffect(() => {
    captureLiteEvent('lite_page_view', { page: 'lite' })
  }, [])

  useEffect(() => {
    const start = Date.now()
    let sent = false
    const flushDuration = (reason: string) => {
      if (sent) return
      sent = true
      captureLiteEvent('lite_leave', {
        page: 'lite',
        reason,
        duration_sec: Math.max(1, Math.round((Date.now() - start) / 1000)),
        signal_count: signalCountRef.current,
      })
    }
    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        flushDuration('hidden')
      }
    }
    const onPageHide = () => flushDuration('pagehide')

    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('pagehide', onPageHide)
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('pagehide', onPageHide)
      flushDuration('unmount')
    }
  }, [])

  // Scroll Restoration
  useEffect(() => {
    const savedScrollPos = localStorage.getItem(`scroll_pos_${location.pathname}`)
    if (savedScrollPos && payload) {
      window.scrollTo(0, parseInt(savedScrollPos, 10))
      // Clear it after restoration to avoid unexpected jumps if user reloads
      // localStorage.removeItem(`scroll_pos_${location.pathname}`)
    }
  }, [payload, location.pathname])

  const handleNavigateToChart = (ticker: string) => {
    localStorage.setItem(`scroll_pos_${location.pathname}`, window.scrollY.toString())
    captureLiteEvent('ticker_chart_click', { ticker })
    navigate(`/lite/chart/${ticker}`)
  }

  useEffect(() => {
    let mounted = true
    fetch('/latest.json', { cache: 'no-store' })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`加载失败: ${res.status}`)
        }
        return res.json() as Promise<LitePayload>
      })
      .then((data) => {
        if (mounted) setPayload(data)
        captureLiteEvent('lite_payload_loaded', {
          signal_count: data.count ?? data.signals?.length ?? 0,
          run_id: data.run_id,
        })
      })
      .catch((err) => {
        if (mounted) setError(err.message || '加载失败')
        captureLiteEvent('lite_payload_load_failed', { message: err.message || '加载失败' })
      })
    return () => {
      mounted = false
    }
  }, [])

  const signals = useMemo(() => payload?.signals ?? [], [payload])

  return (
    <div className="lite-page">
      <header className="lite-header">
        <div className="lite-header-left">
          <a
            href="https://github.com/HKUSTDial/DeepEar"
            target="_blank"
            rel="noreferrer"
            className="lite-logo-link"
            title="DeepEar Project"
          >
            <img
              src="/deepear.svg"
              alt="Logo"
              className="lite-logo"
            />
          </a>
          <div>
            <div className="lite-title">DeepEar Lite</div>
            <div className="lite-subtitle">自动扫描 | 信号关联 | 新闻链接</div>
          </div>
        </div>
        <div className="lite-meta">
          <div>更新时间：{formatTime(payload?.generated_at)}</div>
          <div>信号数：{payload?.count ?? signals.length}</div>
          <div>目前更新频率：{UPDATE_INTERVAL_HOURS} 小时/次</div>
        </div>
      </header>

      {error && <div className="lite-error">{error}</div>}

      <main className="lite-list">
        {signals.length === 0 && !error && (
          <div className="lite-empty">暂无数据，请先生成 latest.json</div>
        )}

        {signals.map((signal, index) => (
          <div key={signal.signal_id || index} className="lite-signal-block">
            <SignalCard
              signal={signal}
              onShowChart={handleNavigateToChart}
              onTickerClick={(ticker, item) =>
                captureLiteEvent('signal_ticker_click', {
                  ticker,
                  signal_id: item.signal_id,
                  signal_title: item.title,
                })
              }
              onSummaryToggle={(expanded, item) =>
                captureLiteEvent('signal_summary_toggle', {
                  expanded,
                  signal_id: item.signal_id,
                })
              }
              onSearchToggle={(expanded, item) =>
                captureLiteEvent('signal_search_toggle', {
                  expanded,
                  signal_id: item.signal_id,
                })
              }
              onSourceClick={(url, item) =>
                captureLiteEvent('signal_source_click', {
                  signal_id: item.signal_id,
                  source_url: url,
                })
              }
            />
          </div>
        ))}

      </main>
      <FeedbackButton />
    </div>
  )
}
