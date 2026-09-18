import { useState } from 'react'

const API_BASE = 'http://localhost:8000'

function MiraChat() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi, I'm Mira. Ask me about any engine's status, or maintenance guidance." },
  ])
  const [input, setInput] = useState('')
  const [asking, setAsking] = useState(false)

  async function send() {
    if (!input.trim()) return
    const question = input
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setAsking(true)
    try {
      const res = await fetch(`${API_BASE}/mira/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'main-widget', message: question }),
      })
      const data = await res.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Could not reach the backend.' }])
    }
    setAsking(false)
  }

  return (
    <>
      <button className="mira-fab" onClick={() => setOpen((o) => !o)}>
        {open ? '✕' : '💬'}
      </button>
      {open && (
        <div className="mira-panel">
          <div className="mira-panel-header">MIRA — Maintenance Assistant</div>
          <div className="mira-panel-log">
            {messages.map((m, i) => (
              <div key={i} className={`mira-msg mira-msg-${m.role}`}>{m.content}</div>
            ))}
            {asking && <div className="mira-msg mira-msg-assistant">Thinking...</div>}
          </div>
          <div className="mira-panel-input-row">
            <input
              className="mira-panel-input"
              placeholder="Ask Mira..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && send()}
            />
            <button className="mira-panel-send" onClick={send} disabled={asking}>Send</button>
          </div>
        </div>
      )}
    </>
  )
}

export default MiraChat