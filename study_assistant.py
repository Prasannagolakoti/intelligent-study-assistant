"""
Study Assistant Module
Coordinates AI/LLM processing for:
1. Extracted Notes Generator (simplified student notes topic-by-topic)
2. AI Study Summaries (clear bullet points with bold keywords)
3. Topic Explanation & Key Points (separate key points section, simple English)
4. Important Topic Priorities (High-Priority Study Topics with key points)
5. Grounded Document Q&A (strict PDF grounding, comparison tables, not-found notices)

Supports Google Gemini and offline Local Demonstration Mode with transparent fallbacks.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

from modules.local_demo import LocalStudyAssistant

# Load .env if present
load_dotenv()


class StudyAssistant:
    """
    Central AI Study Assistant controller tailored for average college students.
    Provides simple, clear, exam-oriented study materials based strictly on the uploaded PDF.
    """

    def __init__(self):
        self.local_engine = LocalStudyAssistant()
        self._gemini_client = None

    def get_api_key(self, custom_key: Optional[str] = None) -> Optional[str]:
        """
        Retrieve API key from user parameter, environment variables, or .env.
        Checks GEMINI_API_KEY and GOOGLE_API_KEY.
        """
        if custom_key and custom_key.strip():
            return custom_key.strip()

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key and key.strip() and not key.strip().startswith("your_"):
            return key.strip()

        return None

    def get_active_provider_info(self, custom_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the current operating mode and active provider status.
        """
        key = self.get_api_key(custom_key)
        if key:
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            return {
                "is_llm_active": True,
                "provider": "Google Gemini",
                "status_badge": "🟢 LLM Connected (Google Gemini)",
                "masked_key": masked,
                "description": "Cloud Generative AI (Gemini Flash) with local fallback safety."
            }
        else:
            return {
                "is_llm_active": False,
                "provider": "Local Demonstration Mode",
                "status_badge": "🟡 Local Demo Mode (Rule-based NLP)",
                "masked_key": None,
                "description": "Offline Extractive NLP and Rule-based algorithms (No API key configured)."
            }

    def _call_gemini(self, prompt: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
        """
        Execute prompt against Google Gemini API with error handling.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            candidate_models = [model_name, "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-2.0-flash"]
            last_err = None

            for m in candidate_models:
                try:
                    model = genai.GenerativeModel(m)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        return response.text.strip()
                except Exception as ex:
                    last_err = ex
                    continue

            if last_err:
                raise last_err
            raise RuntimeError("Empty response received from Gemini API.")
        except Exception as e:
            raise RuntimeError(f"Gemini API invocation error: {str(e)}")

    # -------------------------------------------------------------------------
    # Feature 1: Extracted Notes Generator (Student-Friendly, Topic-by-Topic)
    # -------------------------------------------------------------------------
    def generate_extracted_notes(
        self,
        text: str,
        custom_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract and present the complete relevant study content from the uploaded document
        in simple, easy-to-understand English:
        - Plain language for average students without needing extra explanations.
        - Explains difficult words and technical terms in simple words.
        - Organizes using clear headings, subheadings, and bullet points.
        - Rewrites complex textbook paragraphs into friendly language preserving original meaning.
        - Strictly grounded in the provided document without external facts.
        """
        api_key = self.get_api_key(custom_key)
        if not api_key:
            return self.local_engine.generate_extracted_notes(text)

        prompt = f"""
You are an expert, compassionate college teacher helping average students understand their study material easily.
Extract and rewrite the relevant study content from the provided material into comprehensive, student-friendly Extracted Notes.

CRITICAL INSTRUCTIONS:
1. USE ONLY THE PROVIDED DOCUMENT. Do not invent facts, outside definitions, or external examples.
2. SIMPLIFY COMPLEX LANGUAGE: Do not simply copy-paste dense, difficult textbook paragraphs. Rewrite them in clear, simple English that an average student can easily grasp.
3. EXPLAIN TECHNICAL TERMS: Whenever a difficult or technical word appears, explain its meaning immediately in simple words in parentheses or as bullet points.
4. COMPLETE COVERAGE: Cover all relevant topics and subtopics in the document without unnecessarily skipping explanations.
5. TOPIC-BY-TOPIC STRUCTURE: Organize the content with clear numbered headings, subheadings, and bullet points.
6. BOLD KEYWORDS: Bold all important terms, definitions, formulas, and key concepts.

STUDY MATERIAL:
\"\"\"
{text[:9500]}
\"\"\"

Please respond in JSON format with this exact structure:
{{
  "topics_covered": ["Topic 1", "Topic 2", "Topic 3"],
  "notes_markdown": "Full Markdown text of the notes with # Headings, ## Subheadings, and - **Bold Keywords**: simple bullet explanations.",
  "glossary": [
    {{"term": "Technical Term", "simple_meaning": "Simple explanation in plain English"}}
  ]
}}
"""
        try:
            raw_response = self._call_gemini(prompt, api_key)
            cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
            cleaned_json = re.sub(r'```$', '', cleaned_json, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned_json)

            notes_md = parsed.get("notes_markdown", "")
            if not notes_md.startswith("#"):
                notes_md = f"# 📖 Simplified Extracted Study Notes\n\n> 🤖 **Source**: Google Gemini LLM (Simplified for Students)\n\n" + notes_md

            return {
                "mode": "Google Gemini LLM (gemini-1.5-flash)",
                "notes_markdown": notes_md,
                "topics_covered": parsed.get("topics_covered", []),
                "glossary": parsed.get("glossary", []),
                "word_count": len(re.findall(r'\b\w+\b', notes_md)),
                "is_llm": True
            }
        except Exception as e:
            fallback = self.local_engine.generate_extracted_notes(text)
            fallback["warning"] = f"Cloud LLM API error ({str(e)}). Generated using Local Demonstration Engine."
            fallback["is_llm"] = False
            return fallback

    # -------------------------------------------------------------------------
    # Feature 2: AI Study Summary Generator (Clear Bullet Points & Bold Keywords)
    # -------------------------------------------------------------------------
    def generate_summary(
        self,
        text: str,
        length: str = "Medium",
        custom_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an exam-oriented study summary for average students:
        - In clear bullet points instead of long paragraphs.
        - Highlights keywords, definitions, formulas, concepts, advantages, disadvantages, and differences in bold.
        - Short, meaningful, easy to memorize points.
        - Covers all major topics and important subtopics present in the document.
        - Strictly grounded in the provided document without outside information.
        """
        api_key = self.get_api_key(custom_key)

        if not api_key:
            return self.local_engine.generate_summary(text, length=length)

        prompt = f"""
You are an expert tutor creating an exam revision summary for an average college student.
Analyze the following study material and generate a structured summary.

CRITICAL INSTRUCTIONS:
1. USE CLEAR BULLET POINTS: Generate the summary in clear, easy-to-read bullet points instead of long paragraphs.
2. BOLD IMPORTANT KEYWORDS: Highlight important **keywords**, **definitions**, **formulas**, **concepts**, **advantages**, **disadvantages**, and **differences** using bold text.
3. SIMPLE ENGLISH: Explain every concept in simple, student-friendly English.
4. EXAM-ORIENTED & MEMORABLE: Keep each point short, meaningful, and easy to memorize for exams.
5. COMPLETE COVERAGE: Cover all major topics and important subtopics present in the document.
6. STRICT GROUNDING: Use ONLY the provided document. Never add outside information or external facts.
7. DEPTH LEVEL: {length.upper()}.

STUDY NOTES:
\"\"\"
{text[:9000]}
\"\"\"

Respond ONLY in valid JSON format:
{{
  "executive_summary_bullets": [
    "**Core Theme**: First essential summary bullet with bold keywords.",
    "**Main Principle**: Second key bullet point.",
    "**Key Mechanism**: Third key bullet point."
  ],
  "key_definitions": [
    {{"term": "Term Name", "definition": "Simple, clear definition directly from the notes"}}
  ],
  "core_concepts": ["Concept 1", "Concept 2", "Concept 3"],
  "comparisons_and_differences": [
    "**Difference / Advantage**: Clear comparison or advantage/disadvantage point from notes."
  ],
  "formulas_and_rules": [
    "**Formula / Rule**: Important formula, algorithm, or rule stated in notes."
  ],
  "revision_points": [
    "**Point 1**: High-yield exam revision takeaway.",
    "**Point 2**: Important exam statement in bold."
  ]
}}
"""
        try:
            raw_response = self._call_gemini(prompt, api_key)
            cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
            cleaned_json = re.sub(r'```$', '', cleaned_json, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned_json)

            exec_bullets = parsed.get("executive_summary_bullets", [])
            if not exec_bullets and "executive_summary" in parsed:
                exec_bullets = [f"**Key Point**: {s.strip()}" for s in parsed["executive_summary"].split(". ") if s.strip()]

            exec_summary_str = " ".join(b.replace("**", "") for b in exec_bullets)

            # Build comprehensive Markdown document in bullet points
            md_lines = [
                "> 🤖 **AI Source**: Google Gemini LLM (`gemini-1.5-flash`)",
                f"> 🎯 **Revision Mode**: Student-Friendly Exam Summary ({length.capitalize()} Mode) — Clear bullets with bold keywords.",
                "",
                "### 📌 **Executive Revision Overview (Core Bullets)**",
            ]
            for b in exec_bullets:
                md_lines.append(f"- {b}")
            md_lines.append("")

            md_lines.append("### 🔑 **Core Concepts Identified**")
            for c in parsed.get("core_concepts", []):
                md_lines.append(f"- **{c}**: Core high-yield topic from the study material.")
            md_lines.append("")

            md_lines.append("### 📌 **Key Definitions Explained Simply**")
            for kd in parsed.get("key_definitions", []):
                md_lines.append(f"- **{kd.get('term', '')}**: {kd.get('definition', '')}")
            md_lines.append("")

            comparisons = parsed.get("comparisons_and_differences", [])
            if comparisons:
                md_lines.append("### ⚖️ **Comparisons, Differences & Advantages / Disadvantages**")
                for comp in comparisons:
                    md_lines.append(f"- {comp}")
                md_lines.append("")

            formulas = parsed.get("formulas_and_rules", [])
            if formulas:
                md_lines.append("### 🧮 **Formulas, Rules & Key Statements**")
                for f in formulas:
                    md_lines.append(f"- {f}")
                md_lines.append("")

            md_lines.append("### 📝 **Must-Memorize Exam Revision Points**")
            for bp in parsed.get("revision_points", []):
                md_lines.append(f"- {bp}")

            full_markdown = "\n".join(md_lines)

            return {
                "mode": "Google Gemini LLM (gemini-1.5-flash)",
                "summary_type": length,
                "executive_summary": exec_summary_str,
                "executive_summary_bullets": exec_bullets,
                "key_definitions": parsed.get("key_definitions", []),
                "core_concepts": parsed.get("core_concepts", []),
                "comparisons": comparisons,
                "formulas_and_rules": formulas,
                "bullet_points": parsed.get("revision_points", exec_bullets),
                "full_markdown": full_markdown,
                "is_llm": True
            }

        except Exception as e:
            fallback = self.local_engine.generate_summary(text, length=length)
            fallback["warning"] = f"Cloud LLM API request failed ({str(e)}). Switched to Local Extractive Mode."
            fallback["is_llm"] = False
            return fallback

    # -------------------------------------------------------------------------
    # Feature 3: Topic Explanation & Key Points
    # -------------------------------------------------------------------------
    def explain_topic(
        self,
        topic: str,
        context_text: Optional[str] = None,
        custom_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain a topic with:
        1. Key points identified from the document first (separate section).
        2. Simple English bullet-point explanation breaking down difficult concepts.
        3. Highlighted definitions, keywords, formulas in bold.
        4. Simple example ONLY if available in the document.
        5. Strictly grounded in the provided material without outside content.
        """
        api_key = self.get_api_key(custom_key)

        if not api_key:
            return self.local_engine.explain_topic(topic, context_text=context_text)

        context_segment = f"\nRelevant study context from student notes:\n\"\"\"\n{context_text[:5000]}\n\"\"\"\n" if context_text else ""

        prompt = f"""
You are an expert college tutor explaining an academic topic to an average student.
Topic: "{topic}"
{context_segment}

CRITICAL INSTRUCTIONS:
1. IDENTIFY KEY POINTS FIRST: In the notes, identify the key points related to this topic and present them as a separate list.
2. SIMPLE ENGLISH BREAKDOWN: Explain each key point in simple English using clear bullet points.
3. BREAK DIFFICULT CONCEPTS: Break complex ideas into smaller, understandable parts.
4. BOLD IMPORTANT TERMS: Highlight important definitions, keywords, formulas, and exam-relevant statements in bold.
5. DOCUMENT EXAMPLE ONLY: Provide a simple example ONLY if that example is directly mentioned in the notes. If no example is in the notes, state: "No specific example was provided in the uploaded document for this topic."
6. STRICT GROUNDING: Use only the provided document. Do not include unrelated or external information.

Respond with a JSON object matching this schema:
{{
  "key_points": [
    "**Key Point 1**: First key takeaway from the notes with bold keywords.",
    "**Key Point 2**: Second key takeaway from the notes with bold keywords."
  ],
  "simple_overview": "A clear, intuitive explanation of the topic in plain English.",
  "step_by_step": "1. **Step 1**: Step-by-step breakdown of how the concept works in simple language.\\n2. **Step 2**: ...",
  "terminology": "- **Term**: Simple meaning in plain words.",
  "document_example": "Direct example from the notes, or state that no example exists in the document.",
  "practical_example": "Direct example from the notes, or state that no example exists in the document.",
  "recap": "A punchy 60-second exam memory recap in bullet points."
}}
"""
        try:
            raw_response = self._call_gemini(prompt, api_key)
            cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
            cleaned_json = re.sub(r'```$', '', cleaned_json, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned_json)

            parsed["mode"] = "Google Gemini LLM (gemini-1.5-flash)"
            parsed["topic"] = topic
            parsed["is_llm"] = True
            parsed["grounded_notes_used"] = bool(context_text)

            key_pts = parsed.get("key_points", [])
            if isinstance(key_pts, list):
                parsed["key_points_list"] = key_pts
                parsed["key_points"] = "\n".join(f"- {p}" for p in key_pts)
            else:
                parsed["key_points_list"] = [str(key_pts)]

            return parsed

        except Exception as e:
            fallback = self.local_engine.explain_topic(topic, context_text=context_text)
            fallback["warning"] = f"Cloud LLM API request failed ({str(e)}). Switched to Local Demonstration Mode."
            fallback["is_llm"] = False
            return fallback

    # -------------------------------------------------------------------------
    # Feature 4: Important Topics Extractor with Priority Labels
    # -------------------------------------------------------------------------
    def extract_important_topics(
        self,
        text: str,
        custom_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify only the most important topics and subtopics from the document:
        - Prioritized based on document emphasis (repetition, headings, definitions, formulas, comparisons).
        - Labeled as 'High-Priority Study Topics' based on the document content.
        - Shows key points and a short, simple explanation for every topic.
        - Never claims that a topic will definitely appear in an exam.
        - Strictly grounded in the provided document.
        """
        api_key = self.get_api_key(custom_key)

        if not api_key:
            return self.local_engine.extract_important_topics(text)

        prompt = f"""
Analyze the following study notes and extract the top 6-8 most important academic topics.
Base your priorities strictly on the textual emphasis, headings, definitions, formulas, comparisons, and repeated concepts in the notes.

CRITICAL INSTRUCTIONS:
1. Prioritize topics that are central to the document.
2. Label topics as:
   - "High-Priority Study Topic"
   - "Medium-Priority Study Topic"
   - "Foundational Study Topic"
3. DO NOT claim that a topic will definitely appear in an exam.
4. For every topic, provide:
   - A short, simple explanation in plain English.
   - 2-3 clear key points from the document with bold keywords.
   - The reason why this topic is emphasized in the text.
5. Strictly use the provided material. Do not include external knowledge.

STUDY NOTES:
\"\"\"
{text[:8500]}
\"\"\"

Respond ONLY in JSON format:
[
  {{
    "rank": 1,
    "topic": "Topic Name",
    "priority": "High Priority",
    "priority_label": "High-Priority Study Topic",
    "priority_class": "high",
    "simple_explanation": "1-2 sentence simple explanation in plain English.",
    "key_points": [
      "**Point 1**: Key fact from notes.",
      "**Point 2**: Important definition or rule."
    ],
    "why_important": "Reason based on textual emphasis (repeated concepts, definitions, formulas).",
    "reason": "Detailed reason why this concept is emphasized.",
    "context_snippet": "Relevant phrase or sentence from notes."
  }}
]
"""
        try:
            raw_response = self._call_gemini(prompt, api_key)
            cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
            cleaned_json = re.sub(r'```$', '', cleaned_json, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned_json)

            if isinstance(parsed, list) and len(parsed) > 0:
                for item in parsed:
                    item["is_llm"] = True
                    if "priority_label" not in item:
                        item["priority_label"] = f"{item.get('priority', 'High Priority')} Study Topic"
                    if "key_points" not in item:
                        item["key_points"] = [item.get("reason", "Important topic in notes.")]
                return parsed
            else:
                return self.local_engine.extract_important_topics(text)

        except Exception:
            return self.local_engine.extract_important_topics(text)

    # -------------------------------------------------------------------------
    # Feature 6: Grounded Document Q&A (Strict PDF Grounding)
    # -------------------------------------------------------------------------
    def answer_question(
        self,
        question: str,
        text: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer student questions based strictly on the uploaded text:
        - Explains answers in simple, student-friendly English.
        - Uses bullet points, numbered steps, and comparison tables where appropriate.
        - Highlights keywords and concepts in bold.
        - For 'Explain' -> clear explanation with enough points.
        - For 'Short Notes' -> concise, meaningful notes.
        - For 'Difference Between' -> comparison table.
        - For 'List' -> direct bullet points.
        - For 'Explain in detail' -> covers relevant details from document without external fluff.
        - If not in notes -> strictly states: "This information is not available in the uploaded PDF."
        """
        api_key = self.get_api_key(custom_key)

        if not api_key:
            return self.local_engine.answer_question(question, text)

        history_str = ""
        if chat_history:
            history_str = "\nPrior conversation:\n" + "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]
            )

        prompt = f"""
You are the Intelligent Study Assistant. The student is asking a question about their uploaded study material.

STUDENT'S STUDY NOTES:
\"\"\"
{text[:9000]}
\"\"\"
{history_str}

STUDENT QUESTION: "{question}"

CRITICAL GROUNDING & FORMATTING RULES:
1. USE ONLY THE UPLOADED NOTES. Do not guess, speculate, or import external knowledge.
2. STRICT NOT-FOUND RULE: If the answer is NOT present or directly supported in the notes, respond EXACTLY with:
   "⚠️ **This information is not available in the uploaded PDF.**"
   Do not invent answers or guess.
3. SIMPLE ENGLISH: Explain in plain, accessible language suitable for average college students.
4. BOLD KEYWORDS: Bold all important terms, definitions, formulas, and key concepts.
5. QUESTION-SPECIFIC FORMATTING:
   - If the student asks for "Difference Between" or "Compare", use a clear Markdown comparison table (| Feature | Concept A | Concept B |).
   - If the student asks for "Short Notes", provide concise, bulleted revision notes.
   - If the student asks to "List", provide a clear numbered or bulleted list.
   - If the student asks to "Explain" or "Explain in detail", provide step-by-step clear bullet points covering all relevant details in the document.
"""
        try:
            answer_text = self._call_gemini(prompt, api_key)
            is_not_found = "this information is not available in the uploaded pdf" in answer_text.lower()
            return {
                "mode": "Google Gemini LLM (gemini-1.5-flash)",
                "answer": answer_text,
                "found_in_notes": not is_not_found,
                "confidence": 0.95 if not is_not_found else 0.0,
                "is_llm": True
            }
        except Exception as e:
            fallback = self.local_engine.answer_question(question, text)
            fallback["warning"] = f"Cloud LLM API error: {str(e)}. Answered using Local Extractive Retrieval."
            fallback["is_llm"] = False
            return fallback
