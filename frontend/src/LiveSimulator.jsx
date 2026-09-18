import { useState } from 'react'
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function LiveSimulator({ engines, onEngineUpdated }) {
  const [engineId, setEngineId] = useState('')
  const [severity, setSeverity] = useState(0.3)
  const [running, setRunning] = useState(false)
  const [steps, setSteps] = useState([])

  async function runSimulation() {
    if (!engineId) return
    setRunning(true)
    setSteps([{ label: `Injecting synthetic stress onto Engine ${String(engineId).padStart(3, '0')} sensor readings...`, tone: 'neutral' }])

    try {
      await new Promise((r) => setTimeout(r, 400))
      setSteps((s) => [...s, { label: `Running real forward pass through trained LSTM (50-sample MC-Dropout)...`, tone: 'neutral' }])

      const res = await fetch(`${API_BASE}/engine/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine_id: Number(engineId), severity: Number(severity) }),
      })
      const data = await res.json()

      if (data.error) {
        setSteps((s) => [...s, { label: `Simulation failed: ${data.error}`, tone: 'fail' }])
        setRunning(false)
        return
      }

      setSteps((s) => [...s, {
        label: `Predicted RUL: ${data.predicted_rul} cycles ± ${data.uncertainty_std} (95% CI: ${data.confidence_interval_95[0]}–${data.confidence_interval_95[1]})`,
        tone: data.predicted_rul < 30 ? 'fail' : data.predicted_rul < 70 ? 'flag' : 'pass',
      }])

      const status = data.predicted_rul < 30 ? 'critical' : data.predicted_rul < 70 ? 'warn' : 'safe'
      setSteps((s) => [...s, { label: `FINAL STATUS: ${status.toUpperCase()}`, tone: status === 'critical' ? 'fail' : status === 'warn' ? 'flag' : 'pass' }])

      onEngineUpdated(Number(engineId), {
        predicted_rul: data.predicted_rul,
        uncertainty_std: data.uncertainty_std,
        confidence_interval_95: data.confidence_interval_95,
        status,
      })
    } catch (err) {
      setSteps((s) => [...s, { label: 'Simulation failed — check backend.', tone: 'fail' }])
    }
    setRunning(false)
  }

  return (
    <div className="simulator">
      <div className="simulator-header">
        <span className="live-dot"></span>
        <span className="simulator-title">LIVE FAULT INJECTOR — stress-test a real engine through the actual trained model</span>
      </div>
      <div className="simulator-controls">
        <select className="sim-input" value={engineId} onChange={(e) => setEngineId(e.target.value)}>
          <option value="">Select engine...</option>
          {engines.map((e) => <option key={e.id} value={e.id}>Engine {String(e.id).padStart(3, '0')}</option>)}
        </select>
        <div className="sim-slider-wrap">
          <span className="sim-slider-label">Severity: {(severity * 100).toFixed(0)}%</span>
          <input
            type="range" min="0" max="1" step="0.05"
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
            className="sim-slider"
          />
        </div>
        <button className="sim-run-btn" onClick={runSimulation} disabled={!engineId || running}>
          {running ? 'Running...' : '▶ Inject Fault'}
        </button>
      </div>
      {steps.length > 0 && (
        <div className="sim-log">
          {steps.map((s, i) => <div key={i} className={`sim-step sim-${s.tone}`}>{s.label}</div>)}
        </div>
      )}
    </div>
  )
}

export default LiveSimulator