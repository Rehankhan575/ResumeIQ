"""
app/extras
==========
Modular enhancement utilities:
  - Multi-job description ranking (rank_jds)
  - Skill gap analysis and curated course recommendation (skill_gap)
  - Unconscious bias auditing (bias_check)
  - Local opt-in score performance history (record_history, get_history)
"""

from app.extras.bias import bias_check
from app.extras.history import get_history, record_history
from app.extras.ranking import rank_jds
from app.extras.skill_gap import skill_gap

__all__ = [
    "rank_jds",
    "skill_gap",
    "bias_check",
    "record_history",
    "get_history",
]
