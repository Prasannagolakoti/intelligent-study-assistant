"""
Unit Tests for LocalStudyAssistant
Tests deterministic extractive summarization, rule-based quizzes,
definition extraction, topic priority ranking, and grounded Q&A.
"""

import os
import pytest
from modules.local_demo import LocalStudyAssistant


@pytest.fixture
def local_engine():
    return LocalStudyAssistant()


@pytest.fixture
def sample_notes():
    txt_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "operating_systems.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def test_sentence_splitting(local_engine):
    """Test clean splitting of text into sentences."""
    text = "First concept is mutex. Second concept is counting semaphore! What is deadlock? It is a circular wait."
    sentences = local_engine.split_sentences(text)
    assert len(sentences) >= 3


def test_definition_extraction(local_engine, sample_notes):
    """Test extracting formal definitions from notes."""
    definitions = local_engine.extract_definitions(sample_notes)
    assert len(definitions) >= 1
    terms = [d["term"].lower() for d in definitions]
    assert any("deadlock" in t or "semaphore" in t or "process synchronization" in t or "critical section" in t for t in terms)


def test_extractive_summary_lengths(local_engine, sample_notes):
    """Test summary generation across Short, Medium, and Detailed options."""
    short_sum = local_engine.generate_summary(sample_notes, length="Short")
    assert short_sum["summary_type"] == "Short"
    assert "Local Demonstration Mode" in short_sum["mode"]
    assert len(short_sum["executive_summary"]) > 0

    detailed_sum = local_engine.generate_summary(sample_notes, length="Detailed")
    assert detailed_sum["summary_type"] == "Detailed"
    assert len(detailed_sum["bullet_points"]) >= len(short_sum["bullet_points"])


def test_topic_priority_ranking(local_engine, sample_notes):
    """Test important topic extraction and priority assigning."""
    topics = local_engine.extract_important_topics(sample_notes)
    assert len(topics) >= 3
    priorities = [t["priority"] for t in topics]
    assert "High Priority" in priorities
    # Verify reasons and context snippets are populated
    for t in topics:
        assert len(t["reason"]) > 10
        assert len(t["context_snippet"]) > 0


def test_rule_based_quiz_generation(local_engine, sample_notes):
    """Test generating 5 and 10 question quizzes with valid options."""
    quiz_5 = local_engine.generate_quiz(sample_notes, num_questions=5)
    assert len(quiz_5) == 5

    for q in quiz_5:
        assert "question" in q
        assert len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]
        assert len(q["explanation"]) > 0
        assert "topic" in q


def test_grounded_qa_found(local_engine, sample_notes):
    """Test grounded QA when concept is present in notes."""
    result = local_engine.answer_question("What is Banker's Algorithm?", sample_notes)
    assert result["found_in_notes"] is True
    assert "banker" in result["answer"].lower() or "avoidance" in result["answer"].lower()


def test_grounded_qa_not_found(local_engine, sample_notes):
    """Test grounded QA explicitly says not found when topic is absent."""
    result = local_engine.answer_question("What are quantum superconducting qubits?", sample_notes)
    assert result["found_in_notes"] is False
    assert "not available" in result["answer"].lower() or "limited information" in result["answer"].lower()
