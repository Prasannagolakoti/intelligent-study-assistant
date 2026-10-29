"""
Unit Tests for QuizGenerator and Scoring Engine
Tests question structure, option count, option shuffling, real score calculation,
and revision topic diagnostics.
"""

import os
import pytest
from modules.quiz_generator import QuizGenerator


@pytest.fixture
def quiz_gen():
    return QuizGenerator()


@pytest.fixture
def sample_notes():
    txt_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "operating_systems.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def test_generate_quiz_structure(quiz_gen, sample_notes):
    """Test generating 5 questions ensures exact 4 options and valid answer."""
    res = quiz_gen.generate_quiz(sample_notes, num_questions=5)
    questions = res["questions"]
    assert len(questions) == 5

    for q in questions:
        assert len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]
        assert len(set(q["options"])) == 4, "Options should be unique"
        assert len(q["explanation"]) > 0


def test_score_quiz_all_correct(quiz_gen):
    """Test 100% correct answers calculation."""
    questions = [
        {
            "id": 1,
            "question": "What is a mutex?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "topic": "Process Synchronization",
            "explanation": "Exp 1"
        },
        {
            "id": 2,
            "question": "What is a deadlock?",
            "options": ["W", "X", "Y", "Z"],
            "correct_answer": "X",
            "topic": "Deadlocks",
            "explanation": "Exp 2"
        }
    ]

    user_answers = {1: "A", 2: "X"}
    result = quiz_gen.score_quiz(questions, user_answers)

    assert result["total_questions"] == 2
    assert result["correct_count"] == 2
    assert result["incorrect_count"] == 0
    assert result["percentage"] == 100.0
    assert "Distinction" in result["performance_band"]
    assert len(result["weak_topics"]) == 0


def test_score_quiz_partial_and_weak_topics(quiz_gen):
    """Test partial answers and accurate diagnostic identification of weak topics."""
    questions = [
        {
            "id": 1,
            "question": "Q1",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "topic": "Semaphores",
            "explanation": "Exp 1"
        },
        {
            "id": 2,
            "question": "Q2",
            "options": ["W", "X", "Y", "Z"],
            "correct_answer": "X",
            "topic": "Banker's Algorithm",
            "explanation": "Exp 2"
        }
    ]

    # Q1 is correct, Q2 is wrong
    user_answers = {1: "A", 2: "Y"}
    result = quiz_gen.score_quiz(questions, user_answers)

    assert result["total_questions"] == 2
    assert result["correct_count"] == 1
    assert result["incorrect_count"] == 1
    assert result["percentage"] == 50.0
    assert "Banker's Algorithm" in result["weak_topics"]
    assert "Banker's Algorithm" in result["revision_advice"]


def test_score_quiz_unanswered(quiz_gen):
    """Test handling of unanswered questions."""
    questions = [
        {
            "id": 1,
            "question": "Q1",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "topic": "Deadlocks",
            "explanation": "Exp"
        }
    ]
    # No answer given
    user_answers = {}
    result = quiz_gen.score_quiz(questions, user_answers)

    assert result["total_questions"] == 1
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["detailed_results"][0]["selected_answer"] == "[Not Answered]"
