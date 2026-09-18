function FleetTopology({ engines, onSelectEngine }) {
  const width = 520, height = 420, cx = width / 2, cy = height / 2, radius = 150

  const nodes = engines.map((e, i) => {
    const angle = (i / engines.length) * Math.PI * 2 - Math.PI / 2
    return { ...e, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) }
  })

  const colorFor = (status) => status === 'critical' ? '#E4483C' : status === 'warn' ? '#E8A33D' : '#4FAE7C'

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="topology-svg">
      <defs>
        <radialGradient id="topoGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(232,145,66,0.3)" />
          <stop offset="100%" stopColor="rgba(232,145,66,0)" />
        </radialGradient>
      </defs>

      <circle cx={cx} cy={cy} r={radius * 0.55} fill="none" stroke="#2D3644" strokeWidth="1" strokeDasharray="3,4" />
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#2D3644" strokeWidth="1" strokeDasharray="3,4" />
      <circle cx={cx} cy={cy} r={radius * 1.3} fill="none" stroke="#2D3644" strokeWidth="1" strokeDasharray="3,4" />

      <g className="topology-sweep" style={{ transformOrigin: `${cx}px ${cy}px` }}>
        <path
          d={`M ${cx} ${cy} L ${cx} ${cy - radius * 1.35} A ${radius * 1.35} ${radius * 1.35} 0 0 1 ${cx + radius * 1.35 * Math.sin(Math.PI / 6)} ${cy - radius * 1.35 * Math.cos(Math.PI / 6)} Z`}
          fill="url(#topoGlow)"
        />
      </g>

      {nodes.map((n) => (
        <line key={`edge-${n.id}`} x1={cx} y1={cy} x2={n.x} y2={n.y} stroke="#2D3644" strokeWidth="1.5" />
      ))}

      <circle cx={cx} cy={cy} r="30" fill="#161B22" stroke="#E89142" strokeWidth="2" />
      <text x={cx} y={cy + 4} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#E89142" fontWeight="700">HUB</text>

      {nodes.map((n) => {
        const color = colorFor(n.status)
        return (
                    <g key={n.id} onClick={() => onSelectEngine && onSelectEngine(n)} style={{ cursor: 'pointer' }}>
            <circle cx={n.x} cy={n.y} r="24" fill="#0D1117" stroke={color} strokeWidth="2.5">
              {n.status !== 'safe' && (
                <animate attributeName="r" values="24;27;24" dur={n.status === 'critical' ? '0.9s' : '1.8s'} repeatCount="indefinite" />
              )}
            </circle>
            <circle cx={n.x} cy={n.y} r="24" fill={color} opacity="0.15" />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="10" fill={color} fontWeight="700">{n.predicted_rul.toFixed(0)}</text>
            <text x={n.x} y={n.y + 40} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#7D8896">ENG {String(n.id).padStart(3, '0')}</text>
          </g>
        )
      })}
    </svg>
  )
}

export default FleetTopology