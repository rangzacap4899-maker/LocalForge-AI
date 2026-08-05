import json
import math
import sqlite3
import threading
import urllib.request
from pathlib import Path
from typing import Any

def get_embedding(api_url: str, text: str) -> list[float]:
    payload = {"input": text}
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return []

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class VectorDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT UNIQUE,
                    embedding TEXT,
                    response TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    content TEXT,
                    embedding TEXT
                )
            """)

    def add_to_cache(self, query: str, embedding: list[float], response: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO semantic_cache (query, embedding, response) VALUES (?, ?, ?)",
                (query, json.dumps(embedding), response)
            )

    def search_cache(self, query_embedding: list[float], threshold: float = 0.95) -> str | None:
        if not query_embedding:
            return None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT embedding, response FROM semantic_cache")
            best_score = -1.0
            best_response = None
            for row in cursor:
                try:
                    emb = json.loads(row[0])
                    score = cosine_similarity(query_embedding, emb)
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_response = row[1]
                except:
                    continue
            return best_response

    def add_rag_chunk(self, source: str, content: str, embedding: list[float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO rag_chunks (source, content, embedding) VALUES (?, ?, ?)",
                (source, content, json.dumps(embedding))
            )

    def search_rag(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT source, content, embedding FROM rag_chunks")
            for row in cursor:
                try:
                    emb = json.loads(row[2])
                    score = cosine_similarity(query_embedding, emb)
                    results.append({"source": row[0], "content": row[1], "score": score})
                except:
                    continue
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class TaskPlanner:
    def __init__(self, generate_fn):
        self.generate_fn = generate_fn

    def execute_agentic_workflow(self, prompt: str, on_update) -> str:
        # Step 1: Planning
        on_update("กำลังวางแผน (Planning)...\\n")
        plan_prompt = f"Break down this task into a JSON array of sub-tasks. Output ONLY valid JSON: {prompt}"
        plan_response = self.generate_fn([{"role": "user", "content": plan_prompt}], enable_tools=False, disable_stream=True)
        try:
            # Extract JSON if markdown wrapped
            if "```" in plan_response:
                import re
                match = re.search(r"```(?:json)?\n(.*?)\n```", plan_response, re.S)
                if match:
                    plan_response = match.group(1)
            subtasks = json.loads(plan_response)
            if not isinstance(subtasks, list):
                subtasks = [prompt]
        except:
            subtasks = [prompt] # Fallback
            
        # Step 2: Execution & Evaluation
        results = []
        for i, task in enumerate(subtasks):
            on_update(f"กำลังทำขั้นตอนที่ {i+1}/{len(subtasks)}: {task}\\n")
            task_response = self.generate_fn([{"role": "user", "content": str(task)}], enable_tools=True, disable_stream=True)
            results.append(f"Result of {task}: {task_response}")
            
        # Step 3: Synthesis
        on_update("กำลังสรุปผล (Synthesizing)...\\n")
        final_prompt = f"Based on these results, provide the final answer to: {prompt}\\n\\nResults:\\n" + "\\n".join(results)
        return self.generate_fn([{"role": "user", "content": final_prompt}], enable_tools=False, disable_stream=True)

def estimate_complexity(prompt: str) -> int:
    score = 1
    complex_keywords = ["build", "write", "code", "explain", "analyze", "compare", "system", "architecture"]
    if len(prompt.split()) > 30:
        score += 3
    for word in complex_keywords:
        if word in prompt.lower():
            score += 2
    return min(10, score)
