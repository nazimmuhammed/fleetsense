import { useState, useEffect, useRef } from 'react'
import './index.css'
import GaugeCluster from './GaugeCluster'
import EngineDetail from './EngineDetail'
import MiraChat from './MiraChat'
import FleetDrawer from './FleetDrawer'
import LiveSimulator from './LiveSimulator'

const API_BASE = 'http://localhost:8000'
const FLEET_SIZE = 10
const POLL_INTERVAL_MS = 15000

const TABS = [
  { id: 'topology', label: 'Fleet Topology' },
  { id: 'critical', label: 'Critical Engines' },
  { id: 'schedule', label: 'Fleet Scheduling' },
  { id: 'uncertainty', label: 'Uncertainty Map' },
  { id: 'activity', label: 'Activity Log' },
]

function getStatus(rul, uncertaintyStd) {
  if (rul < 30) return 'critical'
  if (rul < 70 || uncertaintyStd > 25) return 'warn'
  return 'safe'
}

function useCountUp(value, duration = 900) {
  const [display, setDisplay] = useState(0)
  const startRef = useRef(null)
  useEffect(() => {
    if (value == null) return
    startRef.current = null
    let raf
    function step(ts) {
      if (startRef.current === null) startRef.current = ts
      const progress = Math.min((ts - startRef.current) / duration, 1)
      setDisplay(value * (1 - Math.pow(1 - progress, 3)))
      if (progress < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  return display
}

function App() {
  const [engines, setEngines] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedEngine, setSelectedEngine] = useState(null)
  const [activeTab, setActiveTab] = useState(null)
  const [activityLog, setActivityLog] = useState([])
  const [flashIds, setFlashIds] = useState([])
  const prevEnginesRef = useRef({})

  useEffect(() => {
    fetchFleet(true)
    const interval = setInterval(() => fetchFleet(false), POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [])

  function logActivity(message, type = 'info') {
    const time = new Date().toLocaleTimeString()
    setActivityLog((prev) => [{ time, message, type }, ...prev].slice(0, 30))
  }

  async function fetchFleet(isFirstLoad) {
    if (isFirstLoad) setLoading(true)
    const ids = Array.from({ length: FLEET_SIZE }, (_, i) => i + 1)
    const results = await Promise.all(
      ids.map((id) =>
        fetch(`${API_BASE}/engine/${id}/status`)
          .then((r) => r.json())
          .then((data) => ({ ...data, id, status: getStatus(data.predicted_rul, data.uncertainty_std) }))
          .catch(() => ({ id, error: true }))
      )
    )
    const clean = results.filter((e) => !e.error)

        if (!isFirstLoad) {
      const changedIds = []
      clean.forEach((e) => {
        const prevStatus = prevEnginesRef.current[e.id]
        if (prevStatus && prevStatus !== e.status) {
          logActivity(`Engine ${String(e.id).padStart(3, '0')} status changed: ${prevStatus.toUpperCase()} → ${e.status.toUpperCase()}`, e.status)
          changedIds.push(e.id)
        }
      })
      if (changedIds.length === 0) {
        logActivity(`Fleet scan complete — ${clean.length} engines checked, no status changes`, 'info')
      } else {
        setFlashIds(changedIds)
        setTimeout(() => setFlashIds([]), 1500)
      }
    } else {
      logActivity(`Initial fleet scan complete — ${clean.length} engines online`, 'info')
    }

    prevEnginesRef.current = Object.fromEntries(clean.map((e) => [e.id, e.status]))
    setEngines(clean)
    setLoading(false)
  }

    function handleEngineUpdated(engineId, newData) {
    setEngines((prev) => prev.map((e) => (e.id === engineId ? { ...e, ...newData } : e)))
    logActivity(`Engine ${String(engineId).padStart(3, '0')} re-evaluated after fault injection → ${newData.status.toUpperCase()} (${newData.predicted_rul} cycles)`, newData.status)
    setFlashIds([engineId])
    setTimeout(() => setFlashIds([]), 1500)
  }

  const criticalCount = engines.filter((e) => e.status === 'critical').length
  const avgHealth = engines.length ? engines.reduce((sum, e) => sum + e.predicted_rul, 0) / engines.length : 0

  const animatedTotal = useCountUp(engines.length)
  const animatedCritical = useCountUp(criticalCount)
  const animatedAvg = useCountUp(avgHealth)

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"></span>
          <span className="brand-name">FLEETSENSE</span>
        </div>
        <div className="fleet-stats">
          <div className="fleet-stat">
            <span className="fleet-stat-value">{animatedTotal.toFixed(0)}</span>
            <span className="fleet-stat-label">ENGINES MONITORED</span>
          </div>
          <div className="fleet-stat">
            <span className="fleet-stat-value critical">{animatedCritical.toFixed(0)}</span>
            <span className="fleet-stat-label">FLAGGED CRITICAL</span>
          </div>
          <div className="fleet-stat">
            <span className="fleet-stat-value">{animatedAvg.toFixed(0)}</span>
            <span className="fleet-stat-label">AVG FLEET RUL</span>
          </div>
        </div>
      </header>

      <div className="capability-strip">
        {TABS.map((t) => (
          <button key={t.id} className="cap-badge" onClick={() => setActiveTab(t.id)}>{t.label}</button>
        ))}
      </div>

      <LiveSimulator engines={engines} onEngineUpdated={handleEngineUpdated} />

      <main className="hangar-floor">
        <div className="hangar-floor-title">
          <span className="live-dot"></span>
          HANGAR FLOOR — LIVE FLEET STATUS (auto-refresh every 15s)
        </div>
        {loading && <div className="loading-text">Loading fleet status...</div>}
        {!loading && engines.length === 0 && <div className="loading-text">No engine data available — check backend.</div>}
        <div className="gauge-grid">
          {engines.map((e, i) => (
            <GaugeCluster
              key={e.id}
              engineId={e.id}
              predictedRul={e.predicted_rul}
              uncertaintyStd={e.uncertainty_std}
              lowerBound={e.confidence_interval_95[0]}
              upperBound={e.confidence_interval_95[1]}
              status={e.status}
              delay={i * 60}
              flashing={flashIds.includes(e.id)}
              onClick={() => setSelectedEngine(e)}
            />
          ))}
        </div>
      </main>

      <EngineDetail engine={selectedEngine} onClose={() => setSelectedEngine(null)} />
      <FleetDrawer tab={activeTab} onClose={() => setActiveTab(null)} engines={engines} activityLog={activityLog} onSelectEngine={setSelectedEngine} />
      <MiraChat />
    </div>
  )
}

export default App