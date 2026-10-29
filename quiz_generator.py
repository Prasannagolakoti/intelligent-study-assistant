"""
Quiz Generator and Evaluation Engine
Handles:
- Generation of 5, 10, or 15 Multiple Choice Questions (LLM or Rule-based)
- Balanced mix of Question Types:
  1. Definition-based questions
  2. Concept-understanding questions
  3. Logical and application-based questions supported by the document
  4. Comparison and difference-based questions
  5. Formula-based or numerical questions (if present in the document)
- Distinct difficulty tiers: Easy, Medium, Challenging
- 4 options per question with 1 guaranteed correct answer
- Shuffling of question order and option choices
- Short, simple explanation for each question
- Accurate score calculation from real student choices
- Topic revision diagnostics for incorrect answers
"""

import json
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from modules.local_demo import LocalStudyAssistant


class QuizGenerator:
    """
    Manages quiz generation, option randomization, question validation,
    scoring, and revision topic diagnosis for average college students.
    """

    def __init__(self):
        self.local_engine = LocalStudyAssistant()

    def generate_quiz(
        self,
        text: str,
        num_questions: int = 5,
        custom_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple-choice quiz questions based strictly on the uploaded document.
        Provides a balanced mix of:
        - Definition questions
        - Concept-understanding questions
        - Logical & application questions
        - Comparison & difference questions
        - Formula / numerical rule questions (if in notes)
        Difficulty levels: Easy, Medium, Challenging.
        """
        if num_questions not in [5, 10, 15]:
            num_questions = 5

        from modules.study_assistant import StudyAssistant
        assistant = StudyAssistant()
        api_key = assistant.get_api_key(custom_key)

        questions: List[Dict[str, Any]] = []
        is_llm = False
        mode_label = "Local Demonstration Mode (Rule-based NLP)"

        if api_key:
            prompt = f"""
You are an expert college examiner creating an exam revision quiz for an average student.
Based SOLELY on the study notes below, generate exactly {num_questions} multiple-choice questions.

CRITICAL REQUIREMENTS:
1. FOCUS ON IMPORTANT CONCEPTS: Prioritize exam-relevant questions that test important concepts rather than trivial details.
2. BALANCED QUESTION MIX: Provide a balanced mix of:
   - Important definition-based questions (question_type: "Definition")
   - Concept-understanding questions (question_type: "Concept Understanding")
   - Logical and application questions supported by the document (question_type: "Logical Application")
   - Comparison and difference questions (question_type: "Comparison / Difference")
   - Formula-based or numerical questions, if present in the document (question_type: "Formula / Rule")
3. DIFFICULTY TIERS: Include a balanced mix of "Easy", "Medium", and "Challenging" questions.
4. FOUR OPTIONS: Exactly 4 distinct options per MCQ, with exactly one correct answer verbatim/directly supported by the notes.
5. SHORT SIMPLE EXPLANATION: Provide a short, simple explanation for each question explaining why the correct answer is right based on the notes.
6. NO OUTSIDE KNOWLEDGE: Formulate questions requiring ONLY the provided study material. Avoid duplicate or confusing questions.

STUDY NOTES:
\"\"\"
{text[:9000]}
\"\"\"

Respond ONLY with a JSON array:
[
  {{
    "id": 1,
    "question": "Clear, meaningful question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Short, simple explanation citing the notes.",
    "topic": "Core Topic",
    "question_type": "Concept Understanding",
    "difficulty": "Medium"
  }}
]
"""
            try:
                raw_response = assistant._call_gemini(prompt, api_key)
                cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
                cleaned_json = re.sub(r'```$', '', cleaned_json, flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned_json)

                if isinstance(parsed, list) and len(parsed) >= 1:
                    questions = self._validate_and_sanitize_questions(parsed, num_questions)
                    if questions:
                        is_llm = True
                        mode_label = "Google Gemini LLM (gemini-1.5-flash)"
            except Exception:
                pass

        if not questions:
            questions = self.local_engine.generate_quiz(text, num_questions=num_questions)
            is_llm = False
            mode_label = "Local Demonstration Mode (Rule-based NLP)"

        # Randomize options for each question so correct answer isn't predictably indexed
        randomized_questions = []
        for q in questions:
            opts = list(q["options"])
            random.shuffle(opts)
            q_copy = dict(q)
            q_copy["options"] = opts
            if q_copy["correct_answer"] not in opts:
                opts[0] = q_copy["correct_answer"]
                random.shuffle(opts)
                q_copy["options"] = opts
            randomized_questions.append(q_copy)

        return {
            "mode": mode_label,
            "is_llm": is_llm,
            "question_count": len(randomized_questions),
            "questions": randomized_questions,
            "disclaimer": "📌 **Practice Exam Notice**: These questions are generated strictly from the textual emphasis of your uploaded notes for revision and practice. They do not guarantee appearance on actual exams."
        }

    def _validate_and_sanitize_questions(
        self,
        raw_list: List[Dict[str, Any]],
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """
        Validate that every question adheres to strict multiple-choice requirements:
        - Must have a question string
        - Must have exactly 4 options
        - Correct answer must be one of the 4 options
        - Must have explanation, topic tag, question_type, and difficulty
        """
        sanitized = []
        for idx, item in enumerate(raw_list, start=1):
            if not isinstance(item, dict):
                continue

            q_text = item.get("question", "").strip()
            options = item.get("options", [])
            correct = item.get("correct_answer", "").strip()
            explanation = item.get("explanation", "").strip()
            topic = item.get("topic", "General Concept").strip()
            q_type = item.get("question_type", "Concept Understanding").strip()
            diff = item.get("difficulty", "Medium").strip()

            if not q_text or not correct:
                continue

            clean_options = [str(opt).strip() for opt in options if str(opt).strip()]
            if len(clean_options) < 4:
                while len(clean_options) < 4:
                    clean_options.append(f"Alternative Option {len(clean_options)+1}")
            elif len(clean_options) > 4:
                clean_options = clean_options[:4]

            if correct not in clean_options:
                clean_options[0] = correct

            sanitized.append({
                "id": idx,
                "question": q_text,
                "options": clean_options,
                "correct_answer": correct,
                "explanation": explanation or f"The correct response based on your notes is '{correct}'.",
                "topic": topic,
                "question_type": q_type,
                "difficulty": diff
            })

            if len(sanitized) >= num_questions:
                break

        return sanitized

    def score_quiz(
        self,
        questions: List[Dict[str, Any]],
        user_answers: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Evaluate student responses, calculate score percentage, and diagnose
        topics needing additional revision.
        """
        if not questions:
            return {
                "total_questions": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "percentage": 0.0,
                "performance_band": "No Questions",
                "weak_topics": [],
                "detailed_results": [],
                "revision_advice": "No questions were evaluated."
            }

        total = len(questions)
        correct_count = 0
        weak_topics_set = set()
        detailed_results = []

        for q in questions:
            qid = q["id"]
            selected = user_answers.get(qid, "").strip()
            actual_correct = q["correct_answer"].strip()
            is_correct = (selected.lower() == actual_correct.lower()) if selected else False

            if is_correct:
                correct_count += 1
            else:
                weak_topics_set.add(q.get("topic", "Core Concepts"))

            detailed_results.append({
                "id": qid,
                "question": q["question"],
                "selected_answer": selected or "[Not Answered]",
                "correct_answer": actual_correct,
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
                "topic": q.get("topic", "General Concept"),
                "question_type": q.get("question_type", "Concept Understanding"),
                "difficulty": q.get("difficulty", "Medium")
            })

        percentage = round((correct_count / total) * 100, 1)

        # Performance Bands
        if percentage >= 90:
            band = "🌟 Distinction (Mastery of Concepts)"
            advice = "Outstanding performance! You have a solid grasp of these topics. Focus on edge cases and advanced applications."
        elif percentage >= 75:
            band = "🎯 Proficient (Exam-Ready)"
            advice = "Strong performance. Review the few missed topics below to secure top examination grades."
        elif percentage >= 50:
            band = "📚 Developing (Revision Recommended)"
            advice = "Good foundational knowledge. Allocate additional revision time to the weak topics identified below."
        else:
            band = "⚠️ Needs Focused Review"
            advice = "Review the Extracted Notes and AI Summary before attempting this quiz again."

        weak_topics = list(weak_topics_set)
        if weak_topics:
            topic_str = ", ".join(f"'{t}'" for t in weak_topics)
            advice += f"\n\n**Suggested Revision Focus:** Prioritize studying: {topic_str}."

        return {
            "total_questions": total,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "percentage": percentage,
            "performance_band": band,
            "weak_topics": weak_topics,
            "detailed_results": detailed_results,
            "revision_advice": advice
        }
