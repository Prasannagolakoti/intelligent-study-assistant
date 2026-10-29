"""
Intelligent Study Assistant Agent - Mini-Capstone Evaluation Script
Measures and benchmarks key performance indicators:
1. Text Extraction Success Rate
2. Percentage of Quiz Questions with Valid Answers
3. System Response Times (Latency in ms)
4. Quiz Scoring Correctness
5. Grounded Q&A Relevance & Out-of-Corpus Rejection Rate
"""

import os
import time
import json
from typing import Dict, Any, List

from modules.document_processor import DocumentProcessor, DocumentProcessingError
from modules.quiz_generator import QuizGenerator
from modules.local_demo import LocalStudyAssistant
from modules.study_assistant import StudyAssistant


def run_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("  INTELLIGENT STUDY ASSISTANT - CAPSTONE EVALUATION SUITE")
    print("=" * 70)
    
    results = {}
    processor = DocumentProcessor()
    quiz_gen = QuizGenerator()
    local_engine = LocalStudyAssistant()
    assistant = StudyAssistant()

    # -------------------------------------------------------------------------
    # 1. Text Extraction Success Rate
    # -------------------------------------------------------------------------
    print("\n[1/5] Evaluating Text Extraction Success Rate...")
    extraction_test_cases = [
        {"name": "OS Sample PDF", "type": "pdf", "path": "sample_data/operating_systems.pdf", "expect_success": True},
        {"name": "ML Sample PDF", "type": "pdf", "path": "sample_data/machine_learning.pdf", "expect_success": True},
        {"name": "OS Sample TXT", "type": "txt", "path": "sample_data/operating_systems.txt", "expect_success": True},
        {"name": "ML Sample TXT", "type": "txt", "path": "sample_data/machine_learning.txt", "expect_success": True},
        {"name": "Empty File Guard", "type": "bytes", "data": b"", "expect_success": False},
        {"name": "Corrupted PDF Guard", "type": "pdf_bytes", "data": b"%PDF-invalid-bytes", "expect_success": False},
    ]

    extraction_passes = 0
    extraction_times = []

    for tc in extraction_test_cases:
        t0 = time.perf_counter()
        success = False
        try:
            if tc["type"] == "pdf":
                with open(tc["path"], "rb") as f:
                    text, pages = processor.extract_from_pdf(f.read())
                success = (len(text) > 100 and pages >= 1)
            elif tc["type"] == "txt":
                with open(tc["path"], "rb") as f:
                    text = processor.extract_from_txt(f.read())
                success = (len(text) > 100)
            elif tc["type"] == "bytes":
                processor.extract_from_txt(tc["data"])
                success = True
            elif tc["type"] == "pdf_bytes":
                processor.extract_from_pdf(tc["data"])
                success = True
        except DocumentProcessingError:
            success = False

        duration_ms = (time.perf_counter() - t0) * 1000
        extraction_times.append(duration_ms)

        expected = tc["expect_success"]
        passed = (success == expected)
        if passed:
            extraction_passes += 1
        print(f"  - {tc['name']:<22}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.1f}ms)")

    extraction_rate = (extraction_passes / len(extraction_test_cases)) * 100
    results["extraction_success_rate"] = extraction_rate
    results["avg_extraction_time_ms"] = round(sum(extraction_times) / len(extraction_times), 2)
    print(f"  >> Text Extraction Success Rate: {extraction_rate:.1f}%")

    # Load corpus for subsequent tests
    with open("sample_data/operating_systems.txt", "r", encoding="utf-8") as f:
        os_notes = f.read()

    # -------------------------------------------------------------------------
    # 2. Percentage of Quiz Questions with Valid Answers
    # -------------------------------------------------------------------------
    print("\n[2/5] Evaluating Quiz Question Validity...")
    quiz_batches = [5, 10, 15]
    total_questions_generated = 0
    valid_questions_count = 0
    quiz_gen_times = []

    for batch_size in quiz_batches:
        t0 = time.perf_counter()
        quiz_data = quiz_gen.generate_quiz(os_notes, num_questions=batch_size)
        duration_ms = (time.perf_counter() - t0) * 1000
        quiz_gen_times.append(duration_ms)

        questions = quiz_data.get("questions", [])
        total_questions_generated += len(questions)

        for q in questions:
            opts = q.get("options", [])
            correct = q.get("correct_answer", "")
            has_4_options = (len(opts) == 4)
            correct_in_opts = (correct in opts)
            unique_opts = (len(set(opts)) == 4)
            has_explanation = bool(q.get("explanation"))

            if has_4_options and correct_in_opts and unique_opts and has_explanation:
                valid_questions_count += 1

        print(f"  - Batch {batch_size} questions: Generated {len(questions)} in {duration_ms:.1f}ms")

    quiz_validity_rate = (valid_questions_count / total_questions_generated) * 100 if total_questions_generated > 0 else 0
    results["quiz_validity_rate"] = quiz_validity_rate
    results["avg_quiz_gen_time_ms"] = round(sum(quiz_gen_times) / len(quiz_gen_times), 2)
    print(f"  >> Quiz Question Validity Rate: {quiz_validity_rate:.1f}% ({valid_questions_count}/{total_questions_generated} valid)")

    # -------------------------------------------------------------------------
    # 3. Quiz Scoring Correctness
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Quiz Scoring Correctness...")
    scoring_test_cases = [
        {"desc": "All 5 Correct", "answers": {1: "A", 2: "B", 3: "C", 4: "D", 5: "A"}, "expected_score": 5, "expected_pct": 100.0},
        {"desc": "3 of 5 Correct", "answers": {1: "A", 2: "B", 3: "C", 4: "WRONG", 5: "WRONG"}, "expected_score": 3, "expected_pct": 60.0},
        {"desc": "0 of 5 Correct", "answers": {1: "W", 2: "W", 3: "W", 4: "W", 5: "W"}, "expected_score": 0, "expected_pct": 0.0},
        {"desc": "Unanswered", "answers": {}, "expected_score": 0, "expected_pct": 0.0},
    ]

    mock_questions = [
        {"id": 1, "question": "Q1", "options": ["A", "B", "C", "D"], "correct_answer": "A", "topic": "Deadlocks"},
        {"id": 2, "question": "Q2", "options": ["A", "B", "C", "D"], "correct_answer": "B", "topic": "Semaphores"},
        {"id": 3, "question": "Q3", "options": ["A", "B", "C", "D"], "correct_answer": "C", "topic": "Banker's Algorithm"},
        {"id": 4, "question": "Q4", "options": ["A", "B", "C", "D"], "correct_answer": "D", "topic": "Mutex"},
        {"id": 5, "question": "Q5", "options": ["A", "B", "C", "D"], "correct_answer": "A", "topic": "Peterson's Algorithm"},
    ]

    scoring_passes = 0
    for stc in scoring_test_cases:
        score_res = quiz_gen.score_quiz(mock_questions, stc["answers"])
        score_ok = (score_res["correct_count"] == stc["expected_score"])
        pct_ok = (score_res["percentage"] == stc["expected_pct"])
        if score_ok and pct_ok:
            scoring_passes += 1
            print(f"  - {stc['desc']:<18}: PASSED (Score: {score_res['correct_count']}/5, Pct: {score_res['percentage']}%)")
        else:
            print(f"  - {stc['desc']:<18}: FAILED")

    scoring_accuracy = (scoring_passes / len(scoring_test_cases)) * 100
    results["scoring_accuracy_rate"] = scoring_accuracy
    print(f"  >> Quiz Scoring Accuracy: {scoring_accuracy:.1f}%")

    # -------------------------------------------------------------------------
    # 4. Grounded Q&A Relevance & Rejection Rate
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating Grounded Q&A Relevance and Grounding...")
    qa_cases = [
        {"q": "What are the four necessary conditions for deadlock?", "in_notes": True},
        {"q": "How does Banker's Algorithm ensure safety?", "in_notes": True},
        {"q": "What is Peterson's solution?", "in_notes": True},
        {"q": "How do quantum superconductors achieve entanglement?", "in_notes": False},
        {"q": "What are the stock prices of Apple Inc.?", "in_notes": False},
    ]

    qa_passes = 0
    qa_times = []
    for qtc in qa_cases:
        t0 = time.perf_counter()
        qa_res = local_engine.answer_question(qtc["q"], os_notes)
        dur = (time.perf_counter() - t0) * 1000
        qa_times.append(dur)

        found = qa_res.get("found_in_notes", False)
        expected = qtc["in_notes"]
        passed = (found == expected)
        if passed:
            qa_passes += 1
        label = "IN-NOTES" if expected else "OUT-OF-NOTES"
        status = "CORRECT" if passed else "INCORRECT"
        print(f"  - [{label}] '{qtc['q'][:35]}...': {status} ({dur:.1f}ms)")

    qa_accuracy = (qa_passes / len(qa_cases)) * 100
    results["grounded_qa_accuracy"] = qa_accuracy
    results["avg_qa_time_ms"] = round(sum(qa_times) / len(qa_times), 2)
    print(f"  >> Grounded Q&A Accuracy: {qa_accuracy:.1f}%")

    # -------------------------------------------------------------------------
    # 5. Summarization & Topic Extraction Latency
    # -------------------------------------------------------------------------
    print("\n[5/5] Measuring End-to-End Processing Latency...")
    t0 = time.perf_counter()
    sum_res = local_engine.generate_summary(os_notes, length="Medium")
    sum_time_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    top_res = local_engine.extract_important_topics(os_notes)
    top_time_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    exp_res = local_engine.explain_topic("Critical Section Problem", os_notes)
    exp_time_ms = (time.perf_counter() - t0) * 1000

    results["summary_latency_ms"] = round(sum_time_ms, 2)
    results["topic_extraction_latency_ms"] = round(top_time_ms, 2)
    results["topic_explain_latency_ms"] = round(exp_time_ms, 2)

    print(f"  - Extractive Summary Latency : {sum_time_ms:.1f}ms")
    print(f"  - Topic Extraction Latency   : {top_time_ms:.1f}ms")
    print(f"  - Topic Explanation Latency  : {exp_time_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # Summary Table Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                     EVALUATION REPORT SUMMARY")
    print("=" * 70)
    print(f" {'Metric Evaluated':<38} | {'Measured Result':<25}")
    print("-" * 70)
    print(f" {'Text Extraction Success Rate':<38} | {results['extraction_success_rate']:.1f}%")
    print(f" {'Quiz Question Validity Rate':<38} | {results['quiz_validity_rate']:.1f}%")
    print(f" {'Quiz Scoring Correctness':<38} | {results['scoring_accuracy_rate']:.1f}%")
    print(f" {'Grounded Q&A Accuracy':<38} | {results['grounded_qa_accuracy']:.1f}%")
    print(f" {'Avg Extraction Response Time':<38} | {results['avg_extraction_time_ms']} ms")
    print(f" {'Avg Quiz Generation Time':<38} | {results['avg_quiz_gen_time_ms']} ms")
    print(f" {'Avg Grounded Q&A Time':<38} | {results['avg_qa_time_ms']} ms")
    print(f" {'Summary Generation Time':<38} | {results['summary_latency_ms']} ms")
    print("=" * 70)

    # Save to JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Evaluation results exported to evaluation_results.json\n")

    return results


if __name__ == "__main__":
    run_evaluation()
