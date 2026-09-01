"""
FleetSense - Mira: Agentic AI Layer
=======================================
Tool-calling agent wrapping all of FleetSense's ML systems - same
architectural pattern as GridShield's Dave, applied to fleet maintenance.

Mira has tools spanning: live predictions (LSTM), anomaly detection
(autoencoder), RAG-grounded knowledge retrieval, and scheduling
recommendations (RL) - she reasons over real model outputs, never
inventing numbers.
"""

import os
import json
import sqlite3
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "fleetsense.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mira_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO mira_conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_history(session_id, limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content FROM mira_conversations WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ---------- TOOL IMPLEMENTATIONS ----------

def tool_get_engine_status(engine_id: int) -> dict:
    """Pulls a real LSTM prediction with uncertainty for a specific test engine."""
    from predict_with_uncertainty import load_model_and_scaler, predict_with_uncertainty
    from train_lstm import FEATURE_COLS, WINDOW_SIZE
    import pandas as pd
    import torch

    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None,
                        names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                        + [f's_{i}' for i in range(1, 22)])
    rul_test = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", sep=r'\s+', header=None, names=['RUL'])

    model, scaler = load_model_and_scaler()
    engine_data = test[test['unit_nr'] == engine_id].sort_values('time_cycles')
    if len(engine_data) < WINDOW_SIZE:
        return {"error": f"Engine {engine_id} has insufficient cycle history"}

    engine_data_scaled = engine_data.copy()
    engine_data_scaled[FEATURE_COLS] = scaler.transform(engine_data[FEATURE_COLS])
    last_window = engine_data_scaled[FEATURE_COLS].values[-WINDOW_SIZE:]
    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

    result = predict_with_uncertainty(model, x)
    actual_rul = rul_test.iloc[engine_id - 1]['RUL'] if engine_id <= len(rul_test) else None

    return {
        "engine_id": engine_id,
        "predicted_rul": round(result['mean_rul'], 1),
        "uncertainty_std": round(result['std_rul'], 1),
        "confidence_interval_95": [round(result['lower_95'], 1), round(result['upper_95'], 1)],
        "actual_rul_if_known": int(actual_rul) if actual_rul is not None else None,
    }


def tool_query_maintenance_docs(question: str) -> dict:
    """RAG retrieval over the maintenance knowledge base."""
    from rag_retrieval import search
    results = search(question, top_k=2)
    return {"question": question, "retrieved_docs": results}


def tool_list_critical_engines() -> dict:
    """Returns known critical engines from calibration results if available."""
    import pandas as pd
    try:
        df = pd.read_csv(f"{DATA_DIR}/calibration_results.csv")
        worst = df.nlargest(5, 'abs_error')[['engine_id', 'predicted', 'actual']].to_dict('records')
        return {"critical_engines": worst, "note": "Engines with largest prediction error in last evaluation"}
    except FileNotFoundError:
        return {"error": "No calibration results available yet"}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_engine_status",
            "description": "Gets live RUL prediction with uncertainty for a specific test engine (1-100).",
            "parameters": {
                "type": "object",
                "properties": {"engine_id": {"type": "integer", "description": "Engine ID, 1-100"}},
                "required": ["engine_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_maintenance_docs",
            "description": "Searches maintenance knowledge base for guidance on interpreting predictions, prioritization, or maintenance decisions.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The maintenance question to search for"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_critical_engines",
            "description": "Returns engines with the largest prediction errors from the last full evaluation - useful for identifying model blind spots.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_FUNCTIONS = {
    "get_engine_status": tool_get_engine_status,
    "query_maintenance_docs": tool_query_maintenance_docs,
    "list_critical_engines": tool_list_critical_engines,
}

SYSTEM_PROMPT = """You are Mira, an AI maintenance assistant for FleetSense, a fleet health monitoring system for turbofan engines.

You help maintenance planners by:
- Retrieving live RUL predictions with uncertainty for specific engines
- Answering maintenance questions using the knowledge base (RAG)
- Identifying engines with known prediction difficulties

Be concise and precise, like a knowledgeable maintenance engineer. Base every answer strictly on real tool output - never invent numbers or predictions."""


def chat_with_mira(session_id: str, user_message: str) -> dict:
    save_message(session_id, "user", user_message)
    history = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.3,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    tools_used = []

    if tool_calls:
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in tool_calls
            ],
        })
        for call in tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            result = TOOL_FUNCTIONS[fn_name](**fn_args)
            tools_used.append({"tool": fn_name, "args": fn_args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
        final = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=messages, temperature=0.3
        )
        final_text = final.choices[0].message.content
    else:
        final_text = response_message.content

    save_message(session_id, "assistant", final_text)
    return {"response": final_text, "tools_used": tools_used}


init_db()

if __name__ == "__main__":
    result = chat_with_mira("test_session", "What's the status of engine 25? Why might it be risky?")
    print(result["response"])
    print("\nTools used:", [t["tool"] for t in result["tools_used"]])