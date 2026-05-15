"""
SQLite-backed investment thesis tracker.
Stores buy/hold/sell theses with key metrics, assumptions, and kill criteria.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "thesis.db")


class ThesisTracker:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS theses (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT NOT NULL,
                    company          TEXT,
                    recommendation   TEXT NOT NULL,
                    current_price    REAL,
                    target_price     REAL,
                    time_horizon     TEXT,
                    thesis           TEXT NOT NULL,
                    key_assumptions  TEXT,
                    kill_criteria    TEXT,
                    metrics_json     TEXT,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_thesis(
        self,
        ticker: str,
        company: str,
        recommendation: str,
        current_price: float,
        target_price: float,
        time_horizon: str,
        thesis: str,
        key_assumptions: str,
        kill_criteria: str,
        metrics_at_time: Optional[Dict] = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO theses
                    (ticker, company, recommendation, current_price, target_price,
                     time_horizon, thesis, key_assumptions, kill_criteria, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, company, recommendation, current_price, target_price,
                time_horizon, thesis, key_assumptions, kill_criteria,
                json.dumps(metrics_at_time) if metrics_at_time else None,
            ))
            return cursor.lastrowid

    def get_theses(self, ticker: Optional[str] = None) -> List[Dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM theses WHERE ticker = ? ORDER BY created_at DESC", (ticker,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM theses ORDER BY created_at DESC"
                ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["date"] = d["created_at"][:10]
            d["metrics"] = json.loads(d["metrics_json"]) if d.get("metrics_json") else {}
            result.append(d)
        return result

    def get_all_tickers(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM theses ORDER BY ticker").fetchall()
        return [r[0] for r in rows]

    def delete_thesis(self, thesis_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM theses WHERE id = ?", (thesis_id,))
