const POSTHOG_KEY = import.meta.env.VITE_PUBLIC_POSTHOG_KEY as string | undefined
const POSTHOG_HOST = (
  (import.meta.env.VITE_PUBLIC_POSTHOG_HOST as string | undefined) || 'https://us.i.posthog.com'
).replace(/\/+$/, '')

const DISTINCT_ID_KEY = 'deepear_lite_distinct_id'
const SESSION_ID_KEY = 'deepear_lite_session_id'

let initialized = false

const isBrowser = () => typeof window !== 'undefined'

const isEnabled = () => Boolean(POSTHOG_KEY && POSTHOG_HOST)

const getOrCreate = (storage: Storage, key: string) => {
  const existing = storage.getItem(key)
  if (existing) return existing

  const next =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  storage.setItem(key, next)
  return next
}

const getDistinctId = () => getOrCreate(localStorage, DISTINCT_ID_KEY)
const getSessionId = () => getOrCreate(sessionStorage, SESSION_ID_KEY)

export const initAnalytics = () => {
  if (!isBrowser() || initialized) return
  initialized = true

  // Ensure IDs are present so downstream events are attributable.
  if (isEnabled()) {
    getDistinctId()
    getSessionId()
  }
}

export const captureLiteEvent = (
  event: string,
  properties: Record<string, unknown> = {},
) => {
  if (!isBrowser() || !isEnabled()) return

  const pathname = window.location.pathname
  if (!pathname.startsWith('/lite')) return

  const payload = {
    api_key: POSTHOG_KEY,
    event,
    properties: {
      distinct_id: getDistinctId(),
      $session_id: getSessionId(),
      app: 'deepear-lite',
      $current_url: window.location.href,
      $pathname: pathname,
      ...properties,
    },
    timestamp: new Date().toISOString(),
  }

  const body = JSON.stringify(payload)
  const endpoint = `${POSTHOG_HOST}/capture/`

  if (navigator.sendBeacon) {
    const blob = new Blob([body], { type: 'application/json' })
    navigator.sendBeacon(endpoint, blob)
    return
  }

  void fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    keepalive: true,
  }).catch(() => {
    // Swallow analytics errors to avoid impacting UX.
  })
}
