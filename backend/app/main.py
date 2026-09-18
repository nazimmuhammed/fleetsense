"""
FleetSense - FastAPI Backend
================================
Exposes the existing ML systems (LSTM RUL prediction with MC-Dropout
uncertainty, RAG-grounded Mira chat agent, and the PPO scheduling agent)
as real HTTP endpoints, so a frontend has something live to call.

Run from inside backend/app:
    uvicorn main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simulate_service import simulate_degradation

from mira_agent import chat_with_mira, tool_get_engine_status, tool_list_critical_engines
from scheduler_service import get_schedule_recommendation
from anomaly_service import get_anomaly_status
app = FastAPI(title="FleetSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin later
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def health():
    return {"status": "FleetSense API is running"}


@app.get("/engine/{engine_id}/status")
def engine_status(engine_id: int):
    result = tool_get_engine_status(engine_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/engine/{engine_id}/anomaly")
def engine_anomaly(engine_id: int):
    result = get_anomaly_status(engine_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
@app.get("/engines/critical")
def critical_engines():
    return tool_list_critical_engines()


@app.post("/mira/chat")
def mira_chat(payload: ChatRequest):
    return chat_with_mira(payload.session_id, payload.message)
class SimulateRequest(BaseModel):
    engine_id: int
    severity: float


@app.post("/engine/simulate")
def simulate(payload: SimulateRequest):
    result = simulate_degradation(payload.engine_id, payload.severity)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/fleet/schedule")
def fleet_schedule(engine_ids: str):
    """
    Pass engine_ids as a comma-separated string, e.g.:
    /fleet/schedule?engine_ids=1,2,3,4,5,6,7,8,9,10
    """
    try:
        ids = [int(x) for x in engine_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="engine_ids must be a comma-separated list of integers")

    result = get_schedule_recommendation(ids)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
