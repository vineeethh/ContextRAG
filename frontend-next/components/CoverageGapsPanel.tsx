'use client'

import { useEffect, useState } from 'react'
import { getCoverageGapsApi, type CoverageGaps } from '@/lib/api'

type Period = 7 | 30

function timeAgo(iso: string): string {
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000)
  if (mins < 60) return `${Math.max(mins, 1)}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default function CoverageGapsPanel() {
  const [period, setPeriod] = useState<Period>(30)
  const [data, setData] = useState<CoverageGaps | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    getCoverageGapsApi(period)
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false))
  }, [period])

  return (
    <div className="p-3 space-y-3">
      {/* Period toggle */}
      <div className="flex gap-1 bg-gray-100 p-0.5 rounded-lg">
        {([7, 30] as Period[]).map((d) => (
          <button
            key={d}
            onClick={() => setPeriod(d)}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              period === d
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {d} days
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex justify-center py-8">
          <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {error && (
        <p className="text-xs text-red-500 text-center py-4">{error}</p>
      )}

      {data && !loading && data.topics.length === 0 && (
        <p className="text-xs text-gray-400 text-center py-4">
          No coverage gaps — every question found relevant content.
        </p>
      )}

      {data && !loading && data.topics.length > 0 && (
        <ul className="space-y-2">
          {data.topics.map((t) => (
            <li
              key={t.question}
              className="bg-white rounded-xl border border-gray-200 px-3 py-2.5 shadow-sm"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs font-medium text-gray-900">{t.question}</p>
                <span
                  className="shrink-0 text-xs font-semibold text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded tabular-nums"
                  title="Times asked"
                >
                  ×{t.count}
                </span>
              </div>
              {t.examples.length > 0 && (
                <ul className="mt-1.5 space-y-0.5">
                  {t.examples.map((ex) => (
                    <li key={ex} className="text-xs text-gray-500 truncate" title={ex}>
                      “{ex}”
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-1.5 text-xs text-gray-400 tabular-nums">
                score {t.avg_score.toFixed(2)} · {timeAgo(t.last_asked)}
              </p>
            </li>
          ))}
        </ul>
      )}

      <p className="text-xs text-gray-400 text-center px-2">
        Questions whose best match scored below {data?.threshold ?? 0.3}, grouped by topic —
        add docs for these
      </p>
    </div>
  )
}
