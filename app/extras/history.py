"""
app/extras/history.py
=====================
Opt-in local SQLite score history logging.

Stores scoring performance metrics without persisting any résumé text,
candidate names, or contact details.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from app.schemas import ScoreResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    jd_title TEXT,
    total_score REAL NOT NULL,
    grade TEXT NOT NULL,
    semantic_match REAL,
    keyword_coverage REAL,
    section_completeness REAL,
    formatting REAL,
    experience_fit REAL,
    quantified_achievements REAL,
    action_verbs REAL,
    readability REAL,
    contact_completeness REAL
);
"""

INSERT_RECORD_SQL = """
INSERT INTO score_history (
    timestamp,
    jd_title,
    total_score,
    grade,
    semantic_match,
    keyword_coverage,
    section_completeness,
    formatting,
    experience_fit,
    quantified_achievements,
    action_verbs,
    readability,
    contact_completeness
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def record_history(
    score_result: ScoreResult,
    jd_title: str = "Unspecified",
    db_path: Optional[Union[str, Path]] = None,
    opt_in: bool = False,
) -> bool:
    """Record scoring aggregate metrics to local SQLite database.

    Privacy Guarantee:
      Explicit opt-in required (defaults to False).
      Zero persistence of candidate text, names, emails, phones, or bullet points.

    Args:
        score_result: :class:`ScoreResult` from ATS scoring stage.
        jd_title: Target job title.
        db_path: Path to SQLite database file (defaults to data/history.db).
        opt_in: Boolean flag confirming user consent to record history.

    Returns:
        True if record was written, False if opt_in was False or an error occurred.
    """
    if not opt_in:
        logger.debug("History recording skipped: opt_in is False.")
        return False

    if not score_result:
        return False

    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_db.parent.mkdir(parents=True, exist_ok=True)

    comps = score_result.components or {}
    ts = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(str(target_db)) as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_TABLE_SQL)
            cursor.execute(
                INSERT_RECORD_SQL,
                (
                    ts,
                    str(jd_title or "Unspecified"),
                    float(score_result.total),
                    str(score_result.grade),
                    float(comps.get("semantic_match", 0.0)),
                    float(comps.get("keyword_coverage", 0.0)),
                    float(comps.get("section_completeness", 0.0)),
                    float(comps.get("formatting", 0.0)),
                    float(comps.get("experience_fit", 0.0)),
                    float(comps.get("quantified_achievements", 0.0)),
                    float(comps.get("action_verbs", 0.0)),
                    float(comps.get("readability", 0.0)),
                    float(comps.get("contact_completeness", 0.0)),
                ),
            )
            conn.commit()
            return True
    except Exception as exc:
        logger.warning("Failed to record score history: %s", exc)
        return False


def get_history(
    db_path: Optional[Union[str, Path]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Retrieve scoring history records from SQLite database."""
    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_db.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(str(target_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM score_history ORDER BY id DESC LIMIT ?;",
                (limit,),
            )
            for row in cursor.fetchall():
                records.append(dict(row))
    except Exception as exc:
        logger.warning("Failed to read score history: %s", exc)

    return records
