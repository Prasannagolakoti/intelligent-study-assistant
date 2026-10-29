"""
Unit Tests for StudyAssistant and HistoryManager
Tests API key handling, provider status, topic explanations, and study history.
"""

import os
import pytest
from modules.study_assistant import StudyAssistant
from modules.history_manager import HistoryManager


@pytest.fixture
def assistant():
    return StudyAssistant()


@pytest.fixture
def history_mgr(tmp_path):
    temp_file = str(tmp_path / "test_history.json")
    return HistoryManager(storage_path=temp_file)


@pytest.fixture
def sample_notes():
    txt_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "machine_learning.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def test_provider_status_without_key(assistant):
    """Test provider status when no key is provided."""
    info = assistant.get_active_provider_info(custom_key="")
    assert info["is_llm_active"] is False
    assert "Local Demonstration Mode" in info["provider"]
    assert "🟡" in info["status_badge"]


def test_provider_status_with_key(assistant):
    """Test provider status when a mock key is provided."""
    info = assistant.get_active_provider_info(custom_key="AIzaSyDummyTestKeyForTesting123456")
    assert info["is_llm_active"] is True
    assert "Google Gemini" in info["provider"]
    assert "🟢" in info["status_badge"]
    assert info["masked_key"] == "AIza...3456"


def test_explain_topic_structure(assistant, sample_notes):
    """Test 5-part beginner-friendly explanation schema."""
    explanation = assistant.explain_topic("Gradient Descent", context_text=sample_notes)
    assert "simple_overview" in explanation
    assert "step_by_step" in explanation
    assert "terminology" in explanation
    assert "practical_example" in explanation
    assert "recap" in explanation
    assert len(explanation["simple_overview"]) > 20


def test_history_manager_flow(history_mgr):
    """Test saving summary, quiz, topics, and wiping history."""
    # 1. Add Summary
    s_rec = history_mgr.add_summary_record(
        document_title="OS Notes",
        summary_type="Medium",
        executive_summary="Processes execute concurrently.",
        mode="Local Demonstration Mode",
        word_count=500
    )
    assert s_rec["document_title"] == "OS Notes"

    # 2. Add Quiz
    q_rec = history_mgr.add_quiz_record(
        document_title="OS Notes",
        score_data={"total_questions": 5, "correct_count": 4, "percentage": 80.0, "performance_band": "Proficient", "weak_topics": ["Deadlocks"]},
        mode="Local Demonstration Mode"
    )
    assert q_rec["percentage"] == 80.0

    # 3. Add Topics
    t_rec = history_mgr.add_topics_record(
        document_title="OS Notes",
        topics=[{"topic": "Semaphores"}, {"topic": "Deadlocks"}],
        mode="Local Demonstration Mode"
    )
    assert len(t_rec["top_topics"]) == 2

    # 4. Check summary stats
    stats = history_mgr.get_summary_stats()
    assert stats["total_quizzes"] == 1
    assert stats["avg_quiz_score"] == 80.0
    assert stats["total_summaries"] == 1

    # 5. Clear history
    cleared = history_mgr.clear_history()
    assert cleared is True
    assert history_mgr.get_summary_stats()["total_quizzes"] == 0


def test_generate_extracted_notes(assistant, sample_notes):
    """Test generating simplified extracted notes topic-by-topic."""
    result = assistant.generate_extracted_notes(sample_notes)
    assert "notes_markdown" in result
    assert len(result["notes_markdown"]) > 50
    assert "topics_covered" in result
    assert isinstance(result["topics_covered"], list)
    assert result["word_count"] > 0


def test_explain_topic_key_points_first(assistant, sample_notes):
    """Test that key points are extracted and provided for topic explanation."""
    res = assistant.explain_topic("Linear Regression", context_text=sample_notes)
    assert "key_points" in res
    assert "key_points_list" in res
    assert len(res["key_points_list"]) > 0


def test_summary_bullets_structure(assistant, sample_notes):
    """Test bullet-point summary generation with key categories."""
    sum_res = assistant.generate_summary(sample_notes, length="Short")
    assert "executive_summary_bullets" in sum_res
    assert len(sum_res["executive_summary_bullets"]) > 0
    assert "key_definitions" in sum_res
    assert "core_concepts" in sum_res


def test_important_topics_student_fields(assistant, sample_notes):
    """Test that important topics include simple explanations and key points."""
    topics = assistant.extract_important_topics(sample_notes)
    assert len(topics) > 0
    first = topics[0]
    assert "simple_explanation" in first
    assert "key_points" in first
    assert "priority_label" in first
    assert "Study Topic" in first["priority_label"]
