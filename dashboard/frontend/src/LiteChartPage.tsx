import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import type { ChartData, Signal } from './store'
import { KLineChart } from './components/KLineChart'
import { captureLiteEvent } from './analytics'
import { FeedbackButton } from './components/FeedbackButton'
import './LiteDashboard.css'

type LitePayload = {
  generated_at?: string
  run_id?: string
  count?: number
  signals?: Signal[]
  charts?: Record<string, ChartData>
}

const formatTime = (value?: string) => {
  if (!value) return '未知'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN')
}

export const LiteChartPage = () => {
  const { ticker } = useParams()
  const [payload, setPayload] = useState<LitePayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    captureLiteEvent('lite_chart_view', { page: 'lite_chart', ticker })
  }, [ticker])

  useEffect(() => {
    const start = Date.now()
    let sent = false
    const flushDuration = (reason: string) => {
      if (sent) return
      sent = true
      captureLiteEvent('lite_chart_leave', {
        page: 'lite_chart',
        ticker,
        reason,
        duration_sec: Math.max(1, Math.round((Date.now() - start) / 1000)),
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
  }, [ticker])

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
        captureLiteEvent('lite_chart_payload_loaded', {
          ticker,
          signal_count: data.count ?? data.signals?.length ?? 0,
          run_id: data.run_id,
        })
      })
      .catch((err) => {
        if (mounted) setError(err.message || '加载失败')
        captureLiteEvent('lite_chart_payload_load_failed', { ticker, message: err.message || '加载失败' })
      })
    return () => {
      mounted = false
    }
  }, [])

  const chart = useMemo(() => {
    if (!ticker) return undefined
    return payload?.charts?.[ticker]
  }, [payload, ticker])

  const relatedSignals = useMemo(() => {
    if (!ticker) return []
    return (payload?.signals ?? []).filter((s) =>
      s.impact_tickers?.some((t) => String(t.ticker) === String(ticker))
    )
  }, [payload, ticker])

  return (
    <div className="lite-page">
      <header className="lite-header">
        <div>
          <div className="lite-title">K线预测 · {ticker}</div>
          <div className="lite-subtitle">预测K线 + 逻辑解释</div>
        </div>
        <div className="lite-meta">
          <div>更新时间：{formatTime(payload?.generated_at)}</div>
          <div>Run：{payload?.run_id || '未知'}</div>
        </div>
      </header>

      <div className="lite-nav">
        <Link to="/lite">← 返回Lite</Link>
      </div>

      {error && <div className="lite-error">{error}</div>}

      {!chart && !error && (
        <div className="lite-empty">暂无该标的图表数据</div>
      )}

      {chart && (
        <div className="lite-chart-detail">
          <div className="lite-chart-panel">
            <KLineChart data={chart} group={`lite-detail-${ticker}`} />
          </div>
          <div className="lite-explain">
            <div className="lite-section-title">预测解释</div>
            <div className="lite-explain-text">
              {chart.prediction_logic || '暂无解释'}
            </div>
          </div>
        </div>
      )}

      {relatedSignals.length > 0 && (
        <section className="lite-related">
          <div className="lite-section-title">相关信号与来源</div>
          <div className="lite-related-list">
            {relatedSignals.map((signal, idx) => (
              <div key={signal.signal_id || idx} className="lite-related-item">
                <div className="lite-related-title">{signal.title}</div>
                {signal.sources && signal.sources.length > 0 && (
                  <div className="lite-related-links">
                    {signal.sources.map((src, linkIdx) => (
                      <a
                        key={`${signal.signal_id || idx}-src-${linkIdx}`}
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        onClick={() =>
                          captureLiteEvent('lite_related_source_click', {
                            ticker,
                            signal_id: signal.signal_id,
                            source_url: src.url,
                          })
                        }
                      >
                        {src.title}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
      <FeedbackButton />
    </div>
  )
}
