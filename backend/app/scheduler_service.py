"""
FleetSense - Live Scheduling Recommendation Service
=======================================================
Wraps the trained PPO scheduling agent so it can be queried for a live
recommendation across a specific set of real engines, using their actual
LSTM-predicted RUL as the observation - not a fresh random simulation.

IMPORTANT: the observation/action space size (number of engines) MUST match
whatever n_engines the PPO model was originally trained with in
train_rl_scheduler.py. Default assumed here is 10 - verify against your
actual training run before trusting this in a demo.
"""
from stable_baselines3 import PPO
import numpy as np

from mira_agent import tool_get_engine_status

import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rl_scheduler_model")
EXPECTED_FLEET_SIZE = 10  # must match n_engines used during training
MAX_RUL_NORMALIZATION = 150.0  # matches scheduling_env's random init upper bound

_ppo_model = None


def _get_model():
    global _ppo_model
    if _ppo_model is None:
        _ppo_model = PPO.load(MODEL_PATH)
    return _ppo_model


def get_schedule_recommendation(engine_ids: list[int]) -> dict:
    if len(engine_ids) != EXPECTED_FLEET_SIZE:
        return {
            "error": f"This scheduler was trained on exactly {EXPECTED_FLEET_SIZE} engines "
                     f"at a time - got {len(engine_ids)}. Pass exactly {EXPECTED_FLEET_SIZE} engine IDs."
        }

    statuses = [tool_get_engine_status(eid) for eid in engine_ids]
    for s, eid in zip(statuses, engine_ids):
        if "error" in s:
            return {"error": f"Engine {eid}: {s['error']}"}

    ruls = np.array([s["predicted_rul"] for s in statuses], dtype=np.float32)
    obs = np.clip(ruls / MAX_RUL_NORMALIZATION, 0, 1)

    model = _get_model()
    action, _ = model.predict(obs, deterministic=True)

    recommendations = []
    for eid, rul, act in zip(engine_ids, ruls, action):
        recommendations.append({
            "engine_id": eid,
            "predicted_rul": round(float(rul), 1),
            "service_recommended": bool(act),
        })

    return {"fleet_size": EXPECTED_FLEET_SIZE, "recommendations": recommendations}