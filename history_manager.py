"""
Study History Manager Module
Manages student study session history, generated summaries, quiz attempts,
and topic extractions.
Provides session persistence and local file persistence with clear privacy
guarantees and full deletion controls.
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Any

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".study_history.json")


class HistoryManager:
    """
    Manages study activity logs locally.
    Does not collect any personal data or transmit history over the network.
    Data is stored locally in '.study_history.json'.
    """

    def __init__(self, storage_path: str = HISTORY_FILE):
        self.storage_path = storage_path

    def _load_raw_data(self) -> Dict[str, Any]:
        """Load history from local JSON file or return empty structure."""
        if not os.path.exists(self.storage_path):
            return {"summaries": [], "quizzes": [], "topics": []}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        "summaries": data.get("summaries", []),
                        "quizzes": data.get("quizzes", []),
                        "topics": data.get("topics", [])
                    }
        except Exception:
            pass
        return {"summaries": [], "quizzes": [], "topics": []}

    def _save_raw_data(self, data: Dict[str, Any]) -> bool:
        """Save history to local JSON file safely."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def get_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve all recorded study sessions."""
        return self._load_raw_data()

    def add_summary_record(
        self,
        document_title: str,
        summary_type: str,
        executive_summary: str,
        mode: str,
        word_count: int
    ) -> Dict[str, Any]:
        """Record a generated summary."""
        record = {
            "id": f"sum_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "document_title": document_title or "Untitled Study Material",
            "summary_type": summary_type,
            "executive_summary": executive_summary[:300] + ("..." if len(executive_summary) > 300 else ""),
            "mode": mode,
            "word_count": word_count
        }

        data = self._load_raw_data()
        data["summaries"].insert(0, record)
        # Keep maximum 50 records
        data["summaries"] = data["summaries"][:50]
        self._save_raw_data(data)
        return record

    def add_quiz_record(
        self,
        document_title: str,
        score_data: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """Record an interactive quiz attempt and evaluation score."""
        record = {
            "id": f"quiz_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "document_title": document_title or "Untitled Study Material",
            "total_questions": score_data.get("total_questions", 0),
            "correct_count": score_data.get("correct_count", 0),
            "percentage": score_data.get("percentage", 0.0),
            "performance_band": score_data.get("performance_band", ""),
            "weak_topics": score_data.get("weak_topics", []),
            "mode": mode
        }

        data = self._load_raw_data()
        data["quizzes"].insert(0, record)
        data["quizzes"] = data["quizzes"][:50]
        self._save_raw_data(data)
        return record

    def add_topics_record(
        self,
        document_title: str,
        topics: List[Dict[str, Any]],
        mode: str
    ) -> Dict[str, Any]:
        """Record topic extraction results."""
        top_topic_names = [t.get("topic", "") for t in topics[:5]]
        record = {
            "id": f"top_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "document_title": document_title or "Untitled Study Material",
            "top_topics": top_topic_names,
            "topic_count": len(topics),
            "mode": mode
        }

        data = self._load_raw_data()
        data["topics"].insert(0, record)
        data["topics"] = data["topics"][:50]
        self._save_raw_data(data)
        return record

    def clear_history(self) -> bool:
        """Completely wipe study history from local storage."""
        empty_data = {"summaries": [], "quizzes": [], "topics": []}
        return self._save_raw_data(empty_data)

    def export_as_json(self) -> str:
        """Export history formatted as a JSON string for backup."""
        data = self._load_raw_data()
        return json.dumps(data, indent=2, ensure_ascii=False)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate metrics for student dashboard."""
        data = self._load_raw_data()
        quizzes = data.get("quizzes", [])
        summaries = data.get("summaries", [])
        topics = data.get("topics", [])

        total_quizzes = len(quizzes)
        avg_score = round(sum(q.get("percentage", 0.0) for q in quizzes) / total_quizzes, 1) if total_quizzes > 0 else 0.0
        total_summaries = len(summaries)

        return {
            "total_quizzes": total_quizzes,
            "avg_quiz_score": avg_score,
            "total_summaries": total_summaries,
            "total_topics_analyzed": len(topics)
        }
