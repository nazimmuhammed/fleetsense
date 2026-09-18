import { useState, useEffect } from 'react'
import FleetTopology from './FleetTopology'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function UncertaintyMap({ engines }) {
  const w = 480, h = 280, pad = 40
  const maxRul = 350, maxUncertainty = 40
  const xScale = (v) => pad + (v / maxRul) * (w - pad * 2)
  const yScale = (v) => (h - pad) - (v / maxUncertainty) * (h - pad * 2)
  const colorFor = (status) => status === 'critical' ? '#E4483C' : status === 'warn' ? '#E8A33D' : '#4FAE7C'

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="scatter-svg">
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#2D3644" strokeWidth="1" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#2D3644" strokeWidth="1" />
      <text x={w / 2} y={h - 8} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#7D8896">predicted RUL (cycles) →</text>
      <text x={12} y={h / 2} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#7D8896" transform={`rotate(-90 12 ${h / 2})`}>uncertainty →</text>
      {engines.map((e) => (
        <circle key={e.id} cx={xScale(e.predicted_rul)} cy={yScale(e.uncertainty_std)} r="5" fill={colorFor(e.status)} opacity="0.8" />
      ))}
    </svg>
  )
}

function FleetDrawer({ tab, onClose, engines, activityLog, onSelectEngine }) {
  const [criticalData, setCriticalData] = useState(null)
  const [scheduleData, setScheduleData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!tab) return
    setLoading(true)
    if (tab === 'critical') {
      fetch(`${API_BASE}/engines/critical`).then((r) => r.json()).then((d) => { setCriticalData(d); setLoading(false) })
    } else if (tab === 'schedule') {
      const ids = engines.map((e) => e.id).join(',')
      fetch(`${API_BASE}/fleet/schedule?engine_ids=${ids}`).then((r) => r.json()).then((d) => { setScheduleData(d); setLoading(false) })
    } else {
      setLoading(false)
    }
  }, [tab])

  if (!tab) return null

  const titles = {
    critical: 'CRITICAL ENGINES',
    schedule: 'FLEET SCHEDULING',
    uncertainty: 'UNCERTAINTY MAP',
    topology: 'FLEET TOPOLOGY',
    activity: 'ACTIVITY LOG',
  }

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <span className="drawer-title">{titles[tab]}</span>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>
        <div className="drawer-body">
          {loading && <div className="detail-loading">Loading...</div>}

          {!loading && tab === 'critical' && (
            criticalData?.error ? (
              <div className="detail-loading">{criticalData.error}</div>
            ) : (
              <div className="critical-list">
                {criticalData?.critical_engines?.map((e, i) => (
                  <div key={i} className="critical-row">
                    <span className="critical-engine-id">ENGINE {String(e.engine_id).padStart(3, '0')}</span>
                    <span className="critical-values">predicted {e.predicted?.toFixed?.(0) ?? e.predicted} / actual {e.actual}</span>
                  </div>
                ))}
                <div className="detail-explanation" style={{ marginTop: 16 }}>{criticalData?.note}</div>
              </div>
            )
          )}

          {!loading && tab === 'schedule' && (
            <div className="schedule-list">
              {scheduleData?.recommendations?.map((r) => (
                <div key={r.engine_id} className="schedule-row">
                  <span className="schedule-engine-id">ENGINE {String(r.engine_id).padStart(3, '0')}</span>
                  <span className="schedule-rul">{r.predicted_rul.toFixed(0)} cycles</span>
                  <span className={`schedule-tag ${r.service_recommended ? 'recommend' : 'monitor'}`}>
                    {r.service_recommended ? '🔧 SERVICE' : '✓ MONITOR'}
                  </span>
                </div>
              ))}
            </div>
          )}

          {!loading && tab === 'uncertainty' && (
            <>
              <div className="detail-explanation" style={{ marginBottom: 16 }}>
                Every engine's predicted RUL, plotted against the model's own uncertainty (Monte Carlo Dropout, 50 inference passes).
              </div>
              <UncertaintyMap engines={engines} />
              <div className="uncertainty-legend">
                <span className="legend-item"><span className="legend-dot" style={{ background: '#4FAE7C' }}></span>Safe</span>
                <span className="legend-item"><span className="legend-dot" style={{ background: '#E8A33D' }}></span>Warn</span>
                <span className="legend-item"><span className="legend-dot" style={{ background: '#E4483C' }}></span>Critical</span>
              </div>
            </>
          )}

          {!loading && tab === 'topology' && (
            <>
              <div className="detail-explanation" style={{ marginBottom: 16 }}>
                All 10 engines connected to the FleetSense hub. Pulsing rings indicate WARN or CRITICAL status — pulse speed reflects severity.
              </div>
                            <FleetTopology engines={engines} onSelectEngine={onSelectEngine} />
            </>
          )}

          {!loading && tab === 'activity' && (
            <div className="activity-list">
              {activityLog.length === 0 && <div className="detail-loading">No activity yet — waiting for first scan.</div>}
              {activityLog.map((a, i) => (
                <div key={i} className={`activity-row activity-${a.type}`}>
                  <span className="activity-time">{a.time}</span>
                  <span className="activity-message">{a.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default FleetDrawer