import { useState, useEffect } from 'react'
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function EngineDetail({ engine, onClose }) {
  const [explanation, setExplanation] = useState('')
  const [loadingExplanation, setLoadingExplanation] = useState(true)
  const [schedule, setSchedule] = useState(null)
  const [loadingSchedule, setLoadingSchedule] = useState(true)
  const [anomaly, setAnomaly] = useState(null)
  const [loadingAnomaly, setLoadingAnomaly] = useState(true)

  useEffect(() => {
    if (!engine) return
    setExplanation('')
    setSchedule(null)
    setAnomaly(null)
    fetchExplanation()
    fetchSchedule()
    fetchAnomaly()
  }, [engine])

  async function fetchExplanation() {
    setLoadingExplanation(true)
    try {
      const res = await fetch(`${API_BASE}/mira/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: `detail-${engine.id}`,
          message: `Explain engine ${engine.id}'s current status in 2-3 sentences. Why is it at this risk level? Also check whether the anomaly detector agrees with the RUL prediction.`,
        }),
      })
      const data = await res.json()
      setExplanation(data.response)
    } catch (err) {
      setExplanation('Could not reach Mira — check backend.')
    }
    setLoadingExplanation(false)
  }

  async function fetchSchedule() {
    setLoadingSchedule(true)
    try {
      const ids = Array.from({ length: 10 }, (_, i) => i + 1).join(',')
      const res = await fetch(`${API_BASE}/fleet/schedule?engine_ids=${ids}`)
      const data = await res.json()
      const mine = data.recommendations?.find((r) => r.engine_id === engine.id)
      setSchedule(mine || null)
    } catch (err) {
      setSchedule(null)
    }
    setLoadingSchedule(false)
  }

  async function fetchAnomaly() {
    setLoadingAnomaly(true)
    try {
      const res = await fetch(`${API_BASE}/engine/${engine.id}/anomaly`)
      const data = await res.json()
      setAnomaly(data)
    } catch (err) {
      setAnomaly(null)
    }
    setLoadingAnomaly(false)
  }

  if (!engine) return null

  const rulSaysRisky = engine.status !== 'safe'
  const anomalySaysRisky = anomaly?.is_anomalous
  const modelsAgree = !loadingAnomaly && anomaly && (rulSaysRisky === anomalySaysRisky)

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <span className="drawer-title">ENGINE {String(engine.id).padStart(3, '0')}</span>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>
        <div className="drawer-body">
          <div className="detail-stat-row">
            <div className="detail-stat">
              <span className="detail-stat-value">{engine.predicted_rul.toFixed(0)}</span>
              <span className="detail-stat-label">Predicted RUL</span>
            </div>
            <div className="detail-stat">
              <span className="detail-stat-value">± {engine.uncertainty_std.toFixed(0)}</span>
              <span className="detail-stat-label">Uncertainty (std)</span>
            </div>
            <div className="detail-stat">
              <span className="detail-stat-value">{engine.actual_rul_if_known ?? '—'}</span>
              <span className="detail-stat-label">Actual RUL (ground truth)</span>
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-title">DUAL-MODEL CROSS-CHECK</div>
            <div className="cross-check-grid">
              <div className={`cross-check-card ${rulSaysRisky ? 'flag' : 'ok'}`}>
                <div className="cross-check-model">LSTM (supervised)</div>
                <div className="cross-check-verdict">{rulSaysRisky ? 'ELEVATED RISK' : 'HEALTHY'}</div>
                <div className="cross-check-detail">RUL {engine.predicted_rul.toFixed(0)} cycles</div>
              </div>
              <div className={`cross-check-card ${loadingAnomaly ? '' : anomalySaysRisky ? 'flag' : 'ok'}`}>
                <div className="cross-check-model">Autoencoder (unsupervised)</div>
                <div className="cross-check-verdict">
                  {loadingAnomaly ? 'Checking...' : anomalySaysRisky ? 'ANOMALOUS' : 'NORMAL'}
                </div>
                <div className="cross-check-detail">
                  {loadingAnomaly ? '' : `${(anomaly.severity_ratio * 100).toFixed(0)}% of threshold`}
                </div>
              </div>
            </div>
            {!loadingAnomaly && anomaly && (
              <div className={`agreement-banner ${modelsAgree ? 'agree' : 'disagree'}`}>
                {modelsAgree
                  ? '✓ Both independent models agree on this assessment'
                  : '⚠ Models disagree — worth manual inspection. Two independently-trained approaches (supervised RUL vs. unsupervised reconstruction error) are seeing this engine differently.'}
              </div>
            )}
          </div>

          <div className="detail-section">
            <div className="detail-section-title">95% CONFIDENCE INTERVAL</div>
            <div className="detail-ci">{engine.confidence_interval_95[0].toFixed(0)} — {engine.confidence_interval_95[1].toFixed(0)} cycles</div>
          </div>

          <div className="detail-section">
            <div className="detail-section-title">MIRA'S ASSESSMENT</div>
            {loadingExplanation ? (
              <div className="detail-loading">Asking Mira...</div>
            ) : (
              <div className="detail-explanation">{explanation}</div>
            )}
          </div>

          <div className="detail-section">
            <div className="detail-section-title">SCHEDULING RECOMMENDATION</div>
            {loadingSchedule ? (
              <div className="detail-loading">Consulting RL scheduler...</div>
            ) : schedule ? (
              <div className={`schedule-recommendation ${schedule.service_recommended ? 'recommend' : 'monitor'}`}>
                {schedule.service_recommended ? '🔧 SERVICE RECOMMENDED THIS CYCLE' : '✓ NO ACTION NEEDED — MONITOR'}
              </div>
            ) : (
              <div className="detail-loading">Scheduler unavailable.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default EngineDetail