function StatusIcon({ status, size = 14 }) {
  const color = status === 'safe' ? '#4FAE7C' : status === 'warn' ? '#E8A33D' : '#E4483C'
  if (status === 'safe') {
    return (
      <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="7" r="5.5" stroke={color} strokeWidth="1.4" />
        <path d="M4.5 7L6.2 8.7L9.5 5" stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
  if (status === 'warn') {
    return (
      <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
        <path d="M7 1.5L13 12.5H1L7 1.5Z" stroke={color} strokeWidth="1.4" strokeLinejoin="round" />
        <line x1="7" y1="5.5" x2="7" y2="8.5" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
        <circle cx="7" cy="10.3" r="0.6" fill={color} />
      </svg>
    )
  }
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="5.5" stroke={color} strokeWidth="1.4" />
      <line x1="4.5" y1="4.5" x2="9.5" y2="9.5" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
      <line x1="9.5" y1="4.5" x2="4.5" y2="9.5" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}

function GaugeCluster({ engineId, predictedRul, uncertaintyStd, lowerBound, upperBound, status, onClick, delay = 0, flashing = false }) {
  const maxScale = 350
  const size = 200
  const center = size / 2
  const radius = 74
  const startAngle = -220
  const endAngle = 40
  const angleRange = endAngle - startAngle

  function valueToAngle(value) {
    const clamped = Math.max(0, Math.min(maxScale, value))
    return startAngle + (clamped / maxScale) * angleRange
  }

  function polarToCartesian(angleDeg, r) {
    const angleRad = (angleDeg * Math.PI) / 180
    return { x: center + r * Math.cos(angleRad), y: center + r * Math.sin(angleRad) }
  }

  function describeArc(startAng, endAng, r) {
    const start = polarToCartesian(startAng, r)
    const end = polarToCartesian(endAng, r)
    const largeArc = endAng - startAng <= 180 ? 0 : 1
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`
  }

  const statusColor = status === 'critical' ? '#E4483C' : status === 'warn' ? '#E8A33D' : '#4FAE7C'
  const glowId = `glow-${engineId}`

   const needleAngle = valueToAngle(predictedRul)
  const uncertaintyStartAngle = valueToAngle(Math.max(0, lowerBound))
  const uncertaintyEndAngle = valueToAngle(Math.min(maxScale, upperBound))
  const needleTip = polarToCartesian(needleAngle, radius - 10)

  const ticks = [0, 50, 100, 150, 200, 250, 300, 350]

  return (
    <div className={`gauge-cluster ${status} ${flashing ? 'flashing' : ''}`} onClick={onClick} style={{ animationDelay: `${delay}ms` }}>
      <div className="gauge-svg-wrap">
        <svg viewBox={`0 0 ${size} ${size}`} className="gauge-svg">
          <defs>
            <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <path d={describeArc(startAngle, endAngle, radius)} fill="none" stroke="#2D3644" strokeWidth="12" strokeLinecap="round" />

          {ticks.map((t) => {
            const a = valueToAngle(t)
            const p1 = polarToCartesian(a, radius + 9)
            const p2 = polarToCartesian(a, radius + 15)
            return <line key={t} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#4A5568" strokeWidth="1.5" />
          })}

          <path d={describeArc(uncertaintyStartAngle, uncertaintyEndAngle, radius)} fill="none" stroke="#E89142" strokeWidth="12" strokeLinecap="round" opacity="0.4" filter={`url(#${glowId})`} />

          <circle cx={center} cy={center} r={radius - 22} fill="#0D1117" opacity="0.5" />
                    <g>
            <animateTransform
              attributeName="transform"
              type="rotate"
              values={`-0.6 ${center} ${center}; 0.5 ${center} ${center}; -0.4 ${center} ${center}; 0.6 ${center} ${center}; -0.6 ${center} ${center}`}
              dur="0.5s"
              repeatCount="indefinite"
            />
            <line
              x1={center} y1={center}
              x2={needleTip.x} y2={needleTip.y}
              stroke={statusColor}
              strokeWidth="3.5"
              strokeLinecap="round"
              filter={`url(#${glowId})`}
              style={{ transition: 'x2 0.9s ease, y2 0.9s ease, stroke 0.4s ease' }}
            />
          </g>
          
          <circle cx={center} cy={center} r="7" fill={statusColor} filter={`url(#${glowId})`} />

          <text x={center} y={center + 38} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="26" fontWeight="700" fill={statusColor} filter={`url(#${glowId})`}>
            {predictedRul.toFixed(0)}
          </text>
          <text x={center} y={center + 54} textAnchor="middle" fontFamily="Roboto Condensed" fontSize="8.5" letterSpacing="1" fill="#7D8896">
            CYCLES LEFT
          </text>
        </svg>
        <div className="gauge-status-icon-wrap">
          <StatusIcon status={status} />
        </div>
      </div>
      <div className="gauge-label">ENGINE {String(engineId).padStart(3, '0')}</div>
      <div className="gauge-uncertainty">± {uncertaintyStd.toFixed(0)} cycles (95% CI)</div>
      <span className={`gauge-status-tag ${status}`}>{status.toUpperCase()}</span>
    </div>
  )
}

export default GaugeCluster