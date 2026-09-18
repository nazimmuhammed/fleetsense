function MiraAvatar({ talking = false, size = 40 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 60 60" className="mira-avatar-svg">
      <defs>
        <radialGradient id="miraFaceGrad" cx="35%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#FFC98A" />
          <stop offset="100%" stopColor="#E89142" />
        </radialGradient>
      </defs>

      <circle cx="30" cy="30" r="26" fill="url(#miraFaceGrad)" className="mira-avatar-head" />

      {/* Eyes - blink via CSS animation on scaleY */}
      <g className="mira-eye-group">
        <ellipse cx="21" cy="27" rx="3.2" ry="4" fill="#161B22" className="mira-eye mira-eye-left" />
        <ellipse cx="39" cy="27" rx="3.2" ry="4" fill="#161B22" className="mira-eye mira-eye-right" />
      </g>

      {/* Mouth - shape changes depending on talking state */}
      {talking ? (
        <ellipse cx="30" cy="40" rx="6" ry="4" fill="#161B22" className="mira-mouth-talking" />
      ) : (
        <path d="M23 39 Q30 44 37 39" stroke="#161B22" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      )}

      {/* Little antenna / signal dot to keep the "AI assistant" read */}
      <line x1="30" y1="4" x2="30" y2="10" stroke="#E89142" strokeWidth="2" />
      <circle cx="30" cy="4" r="3" fill="#4FAE7C" className="mira-signal-dot" />
    </svg>
  )
}

export default MiraAvatar