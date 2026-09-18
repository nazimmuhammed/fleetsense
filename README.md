# ✈️ FleetSense

### Predictive maintenance & fleet health intelligence for turbofan engines

> *A compromised engine doesn't fail in one clean step. It degrades quietly, across hundreds of cycles, until a technician finds out too late. FleetSense watches continuously — and tells you, in plain language, exactly why it's worried.*

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)
**Status:** 🟢 Fully live — 4 real trained models, wired end-to-end

---

## 🧨 The Problem

```
🛫 Engines degrade gradually across their operational life
🔧 Standard maintenance is either too early (wasted capacity)
     or too late (unplanned failure — the worst outcome)
📊 A single risk number tells you WHAT, never WHY, and never
     how confident the model actually is
```

Predictive maintenance done wrong is just a fancier way to be surprised. FleetSense is built around a different principle: **never trust one model, one number, or one signal alone.**

---

## 💡 The Approach — Four Independent ML Systems, Not One

```
                    📡  Live sensor readings (test engine)
                              │
              ┌───────────────┼───────────────┐
              │                                │
   🧠 LSTM (supervised)              🔍 AUTOENCODER (unsupervised)
   Predicts Remaining Useful         Trained ONLY on healthy data.
   Life directly, with Monte         Flags anomalous sensor patterns
   Carlo Dropout uncertainty          via reconstruction error —
   (50 passes → risk + confidence)    no RUL labels needed at all.
              │                                │
              └───────────────┬───────────────┘
                              │
                    ⚖️  DO THE MODELS AGREE?
                    Two independently-trained approaches
                    cross-checking the same engine is
                    itself a meaningful signal.
                              │
              ┌───────────────┼───────────────┐
              │                                │
   🎯 RL SCHEDULER (PPO)              🤖 MIRA (tool-calling agent)
   Learns WHEN to service an          Reasons over real model
   engine under real technician       outputs via RAG + function-
   capacity constraints —             calling. Never invents a
   not just IF it's at risk.          number it can't retrieve.
```

---

## 🎛️ What's Actually Inside

| 🧩 System | Paradigm | What it does |
|---|---|---|
| 🧠 **LSTM + MC-Dropout** | Supervised, regression | Predicts Remaining Useful Life directly from a rolling window of sensor readings. Dropout is kept **active** at inference time — 50 forward passes, and the spread across them *is* the uncertainty estimate. |
| 🔍 **Autoencoder** | Unsupervised, anomaly detection | Trained exclusively on each engine's early-life (healthy) cycles. Reconstruction error spikes when fed degraded data — validated to rise from early-life to late-life across all 100 engines. |
| 🎯 **PPO Scheduler** | Reinforcement learning | A custom Gymnasium environment simulating N engines competing for K technicians. Learns to prioritize genuinely at-risk engines over wasting capacity on healthy ones — real reward shaping, not a lookup table. |
| 🤖 **Mira** | Agentic LLM (Groq, tool-calling) | Decides for itself which tool to call — live prediction, anomaly check, or knowledge-base retrieval — before answering. Grounded strictly in real tool output. |
| ⚡ **Live Fault Injector** | Simulation | Extrapolates an engine's **own already-observed sensor trend** further forward, scaled by a severity slider — then runs a genuine forward pass through the trained LSTM. Not random noise: an honest amplification of a real trajectory. |

---

## 🖥️ Live Dashboard

```
┌───────────────────────────────────────────────────────────┐
│  ✈️ FLEETSENSE      10 engines · 0 critical · 94 avg RUL   │
├───────────────────────────────────────────────────────────┤
│ [Fleet Topology][Critical Engines][Scheduling][Uncertainty]│
├───────────────────────────────────────────────────────────┤
│  🔴 LIVE FAULT INJECTOR — stress-test a real engine         │
├───────────────────────────────────────────────────────────┤
│   🎛️  🎛️  🎛️  🎛️  🎛️  🎛️        ← animated gauge clusters, │
│   🎛️  🎛️  🎛️  🎛️                  needle sweeps + tremor    │
└───────────────────────────────────────────────────────────┘
```

Click any gauge → drawer opens showing the **LSTM vs. Autoencoder cross-check** for that specific engine, Mira's plain-language assessment, and the RL scheduler's real recommendation — all fetched live, nothing precomputed.

---

## ⚖️ Dual-Model Cross-Check — Why It Matters

A single model can be confidently wrong. Two independently-trained models — one supervised, one unsupervised, learned from the same data in completely different ways — agreeing on an engine is much stronger evidence than either one alone. When they **disagree**, that disagreement is itself flagged as worth a human's attention, rather than silently averaged away.

---

## 🐛 What Broke (and what it taught me)

| Issue | Root cause | Lesson |
|---|---|---|
| Backend crashed on import | Relative paths (`"data"`) resolved differently depending on which folder Python was launched from | Compute paths relative to the file's own location, not the working directory |
| Scheduler 500 error | Same relative-path bug, in a fourth file | Once is a mistake; check every module that touches disk paths |
| Fault injection made RUL go up at max severity | Random Gaussian noise across sensors doesn't correspond to real degradation — different sensors trend in different directions | Extrapolate the engine's **own observed trend**, don't invent one |

---

## 🛠️ Tech Stack

`FastAPI` · `PyTorch` · `scikit-learn` · `Stable-Baselines3` (PPO) · `Gymnasium` · `Groq` (tool-calling LLM) · `sentence-transformers` + `FAISS` (RAG) · `React` (Vite) · Hand-rolled SVG gauges and radar topology (no charting library — kept the aviation-instrument look authentic)

---

## 🚀 Running it locally

```bash
# Backend
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir app

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` 🎉

---

<p align="center">
Built on the NASA C-MAPSS FD001 dataset · a real, established predictive-maintenance benchmark
</p>
