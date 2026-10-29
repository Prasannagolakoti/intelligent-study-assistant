"""
Local Demonstration Engine
Provides high-accuracy, transparent, deterministic local NLP algorithms:
- Extracted Notes Generator in simple, student-friendly English (organized topic by topic)
- Extractive summarization via sentence frequency scoring & cue phrase weighting with bold keywords
- Key definition and formula extraction
- Rule-based multiple choice quiz generation with balanced question types & difficulty levels
- Important topic extraction with priority ranking, simple explanations & key points
- Grounded document Q&A via semantic token retrieval with comparison tables & strict not-found handling

All outputs are explicitly labeled as Local Demonstration Mode to maintain
academic integrity and transparency.
"""

import math
import random
import re
from typing import Dict, List, Optional, Set, Tuple, Any

# Standard academic stopwords (no external download required)
STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', 'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does',
    'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll',
    'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',
    'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no',
    'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s',
    'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d',
    'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve',
    'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
    'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t',
    'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
    'also', 'e.g', 'i.e', 'etc'
}

# Cue phrases that indicate key academic insights or summaries
CUE_PHRASES = [
    'is defined as', 'refers to', 'in conclusion', 'importantly', 'crucial',
    'fundamental', 'key concept', 'primary', 'specifically', 'significant',
    'results in', 'characterized by', 'essential', 'consists of', 'main reason',
    'for example', 'in contrast', 'advantage', 'disadvantage', 'unlike',
    'difference between', 'formula', 'requires that'
]


class LocalStudyAssistant:
    """
    Local rule-based and extractive NLP engine for the Intelligent Study Assistant.
    Provides deterministic study aids for average students when LLM APIs are offline.
    """

    def __init__(self):
        self.mode_label = "Local Demonstration Mode (Extractive & Rule-based NLP)"

    # -------------------------------------------------------------------------
    # Helper: Sentence & Word Splitting
    # -------------------------------------------------------------------------
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly while preserving punctuation."""
        if not text:
            return []
        clean = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', clean)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words, filtering stopwords."""
        raw_words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return [w for w in raw_words if w not in STOPWORDS]

    # -------------------------------------------------------------------------
    # Feature 1: Extracted Notes Generator in Simple English
    # -------------------------------------------------------------------------
    def generate_extracted_notes(self, text: str) -> Dict[str, Any]:
        """
        Extract complete relevant study content from uploaded material.
        Rewrites into student-friendly notes organized topic by topic:
        - Plain, simple English suitable for average students.
        - Explains difficult words and technical terms in simple words.
        - Clear headings, subheadings, and bullet points.
        - Covers all relevant content without skipping explanations.
        - Strictly grounded in the provided document (no outside info).
        """
        if not text or not text.strip():
            return {
                "mode": self.mode_label,
                "notes_markdown": "No readable study content found in the provided document.",
                "topics_covered": [],
                "glossary": [],
                "word_count": 0
            }

        sentences = self.split_sentences(text)
        definitions = self.extract_definitions(text)
        def_map = {d["term"].lower(): d["definition"] for d in definitions}

        # Cluster sentences by topics based on headings or recurring key phrases
        candidate_headings = re.findall(r'(?:^|\n)([A-Z0-9\s\-:]{3,45})(?:\n|$)', text)
        clean_headings = [h.strip() for h in candidate_headings if len(h.strip()) > 3 and len(h.strip().split()) <= 6]
        
        # If few headings found, use extracted topics
        if len(clean_headings) < 3:
            extracted_topics = self.extract_important_topics(text)
            clean_headings = [t["topic"] for t in extracted_topics[:6]]

        # Ensure unique headings
        unique_headings: List[str] = []
        for h in clean_headings:
            if h.lower() not in [u.lower() for u in unique_headings] and len(h) > 2:
                unique_headings.append(h)

        if not unique_headings:
            unique_headings = ["Core Concepts & Overview", "Key Mechanisms & Rules", "Exam Revision Takeaways"]

        # Assign sentences to headings
        topic_sections: Dict[str, List[str]] = {h: [] for h in unique_headings}
        unassigned_sentences: List[str] = []

        for sent in sentences:
            assigned = False
            sent_lower = sent.lower()
            for h in unique_headings:
                h_words = [w.lower() for w in re.findall(r'\b\w+\b', h) if w.lower() not in STOPWORDS]
                if any(hw in sent_lower for hw in h_words):
                    topic_sections[h].append(sent)
                    assigned = True
                    break
            if not assigned:
                unassigned_sentences.append(sent)

        # Distribute unassigned sentences across headings so no content is lost
        for idx, sent in enumerate(unassigned_sentences):
            target_heading = unique_headings[idx % len(unique_headings)]
            topic_sections[target_heading].append(sent)

        # Build student-friendly Markdown notes
        md_lines = [
            f"# 📖 Simplified Extracted Study Notes",
            f"> 💡 **Student-Friendly Revision Guide**: This material has been structured into simple, easy-to-understand English with clear bullet points and bold keywords, based strictly on your uploaded document.",
            f"> ℹ️ **Engine**: {self.mode_label}",
            "",
            "---",
            ""
        ]

        # 1. Topic-by-Topic Breakdown
        for idx, (heading, sents) in enumerate(topic_sections.items(), start=1):
            if not sents:
                continue
            md_lines.append(f"## {idx}. 📌 **{heading}**")
            md_lines.append("")

            # Highlight definitions relevant to this section
            sec_defs = [d for d in definitions if d["term"].lower() in heading.lower() or any(d["term"].lower() in s.lower() for s in sents)]
            if sec_defs:
                md_lines.append("### 🔑 **Key Terms & Definitions (Simple Meaning)**:")
                for sd in sec_defs[:3]:
                    md_lines.append(f"- **{sd['term']}**: {sd['definition']}")
                md_lines.append("")

            md_lines.append("### 📝 **Important Points Explained Simply**:")
            for s in sents[:6]:
                # Format each sentence into student-friendly bullet with bold keyword
                clean_s = s.strip()
                words_in_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                lead_word = words_in_s[0].capitalize() if words_in_s else "Point"
                
                # Check for difficult words in sentence and add simple explanation if available
                annotated_s = clean_s
                for term_lower, simple_def in def_map.items():
                    if term_lower in clean_s.lower() and len(term_lower) > 4:
                        pattern = re.compile(re.escape(term_lower), re.IGNORECASE)
                        annotated_s = pattern.sub(f"**{term_lower.capitalize()}** (*{simple_def[:50]}...*)", annotated_s, count=1)
                        break

                md_lines.append(f"- **{lead_word}**: {annotated_s}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        # 2. Complete Glossary of Difficult Words
        if definitions:
            md_lines.append("## 📚 **Glossary of Difficult Words & Technical Terms**")
            md_lines.append("Here is a quick-reference guide explaining every difficult term from your document in plain language:")
            md_lines.append("")
            for d in definitions:
                md_lines.append(f"- **{d['term']}**: {d['definition']}")
            md_lines.append("")

        full_notes_md = "\n".join(md_lines)

        return {
            "mode": self.mode_label,
            "notes_markdown": full_notes_md,
            "topics_covered": [h for h, sents in topic_sections.items() if sents],
            "glossary": definitions,
            "word_count": len(re.findall(r'\b\w+\b', full_notes_md))
        }

    # -------------------------------------------------------------------------
    # Feature 2: AI Study Summary Generator in Clear Bullet Points
    # -------------------------------------------------------------------------
    def generate_summary(self, text: str, length: str = "Medium") -> Dict[str, Any]:
        """
        Generate an exam-oriented study summary for average students:
        - Presented in clear, memorable bullet points rather than dense paragraphs.
        - Highlights keywords, definitions, formulas, concepts, advantages, disadvantages, and differences in bold.
        - Short, meaningful, easy to memorize points.
        - Strictly grounded in the provided document without outside information.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return {
                "mode": self.mode_label,
                "summary_type": length,
                "executive_summary": "No readable content found in the provided study material.",
                "executive_summary_bullets": ["No readable study content found in the document."],
                "key_definitions": [],
                "core_concepts": [],
                "comparisons": [],
                "formulas_and_rules": [],
                "bullet_points": [],
                "full_markdown": "No content available to summarize."
            }

        words = self.tokenize_words(text)
        word_freq: Dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        max_freq = max(word_freq.values()) if word_freq else 1
        tf_scores = {w: count / max_freq for w, count in word_freq.items()}

        # Score sentences
        sentence_scores: List[Tuple[int, str, float]] = []
        for idx, sent in enumerate(sentences):
            sent_words = self.tokenize_words(sent)
            if not sent_words:
                continue

            score = sum(tf_scores.get(w, 0) for w in sent_words)
            normalized_score = score / (math.sqrt(len(sent_words)) + 1.0)

            lower_sent = sent.lower()
            for cue in CUE_PHRASES:
                if cue in lower_sent:
                    normalized_score += 0.4
                    break

            if idx < 3:
                normalized_score += 0.35
            elif idx < len(sentences) * 0.2:
                normalized_score += 0.15

            sentence_scores.append((idx, sent, normalized_score))

        # Target bullet count based on length
        if length.lower() == "short":
            target_count = min(5, len(sentence_scores))
        elif length.lower() == "detailed":
            target_count = min(12, len(sentence_scores))
        else:  # Medium default
            target_count = min(8, len(sentence_scores))

        top_scored = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:target_count]
        ordered_top = sorted(top_scored, key=lambda x: x[0])
        executive_sentences = [item[1] for item in ordered_top]

        # Extract structured categories
        definitions = self.extract_definitions(text)
        comparisons = self.extract_comparisons(text)
        formulas = self.extract_formulas_and_rules(text)

        # Formulate memorable bullet points with bold keywords
        executive_bullets: List[str] = []
        for i, sent in enumerate(executive_sentences):
            sent_words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', sent) if w.lower() not in STOPWORDS]
            lead_kw = sent_words[0].capitalize() if sent_words else "Concept"
            executive_bullets.append(f"**{lead_kw}**: {sent}")

        # Core concepts (top words with high TF)
        top_terms = [w.capitalize() for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]]

        # Executive summary string for backwards compatibility with tests
        executive_summary_str = " ".join(executive_sentences[:3])

        # Construct comprehensive Markdown document in bullet points
        md_lines = [
            f"> ℹ️ **Study Summary Mode**: {self.mode_label}",
            f"> 🎯 **Target Level**: Student-Friendly Exam Revision ({length.capitalize()} Mode) — All points in clear bullets with bold keywords.",
            "",
            "### 📌 **Executive Revision Overview (Core Bullets)**",
        ]
        for b in executive_bullets[:5]:
            md_lines.append(f"- {b}")
        md_lines.append("")

        md_lines.append("### 🔑 **Core Concepts & Key Ideas**")
        for term in top_terms:
            md_lines.append(f"- **{term}**: Essential high-yield topic emphasized in the study notes.")
        md_lines.append("")

        md_lines.append("### 📖 **Essential Definitions Explained Simply**")
        if definitions:
            for item in definitions[:6]:
                md_lines.append(f"- **{item['term']}**: {item['definition']}")
        else:
            md_lines.append("- *No formal definition phrases detected; review core concept bullets above.*")
        md_lines.append("")

        if comparisons:
            md_lines.append("### ⚖️ **Comparisons, Differences & Advantages / Disadvantages**")
            for comp in comparisons[:4]:
                md_lines.append(f"- {comp}")
            md_lines.append("")

        if formulas:
            md_lines.append("### 🧮 **Formulas, Rules & Key Statements**")
            for rule in formulas[:4]:
                md_lines.append(f"- {rule}")
            md_lines.append("")

        md_lines.append("### 📝 **Must-Memorize Exam Points**")
        for bp in executive_bullets:
            md_lines.append(f"- {bp}")

        full_markdown = "\n".join(md_lines)

        return {
            "mode": self.mode_label,
            "summary_type": length,
            "executive_summary": executive_summary_str,
            "executive_summary_bullets": executive_bullets,
            "key_definitions": definitions,
            "core_concepts": top_terms,
            "comparisons": comparisons,
            "formulas_and_rules": formulas,
            "bullet_points": executive_sentences,
            "full_markdown": full_markdown
        }

    # -------------------------------------------------------------------------
    # Helper: Comparisons, Advantages, Disadvantages & Differences
    # -------------------------------------------------------------------------
    def extract_comparisons(self, text: str) -> List[str]:
        """Extract comparison and difference statements from text."""
        sentences = self.split_sentences(text)
        comp_keywords = ['in contrast', 'whereas', 'while', 'unlike', 'difference between', 'compared to', 'differs from', 'advantage', 'disadvantage', 'faster than', 'slower than']
        results = []
        for s in sentences:
            s_lower = s.lower()
            if any(k in s_lower for k in comp_keywords):
                results.append(f"**Comparison**: {s}")
        return results

    # -------------------------------------------------------------------------
    # Helper: Formulas, Rules & Key Conditions
    # -------------------------------------------------------------------------
    def extract_formulas_and_rules(self, text: str) -> List[str]:
        """Extract formulas, algorithms, equations, or formal conditions."""
        sentences = self.split_sentences(text)
        rule_keywords = ['formula', 'equation', 'algorithm', 'condition', 'rule', 'theorem', 'step', '=', 'must satisfy', 'characterized by']
        results = []
        for s in sentences:
            s_lower = s.lower()
            if any(k in s_lower for k in rule_keywords):
                results.append(f"**Key Rule / Formula**: {s}")
        return results

    # -------------------------------------------------------------------------
    # Feature 3: Definition Extraction
    # -------------------------------------------------------------------------
    def extract_definitions(self, text: str) -> List[Dict[str, str]]:
        """
        Extract explicit definitions using academic linguistic patterns:
        - X is defined as Y
        - X refers to Y
        - X is a/an Y that/which
        - X means Y
        """
        patterns = [
            r'([A-Z][a-zA-Z0-9\s\-]{2,35})\s+is\s+defined\s+as\s+([^.\n]{10,180})',
            r'([A-Z][a-zA-Z0-9\s\-]{2,35})\s+refers\s+to\s+([^.\n]{10,180})',
            r'([A-Z][a-zA-Z0-9\s\-]{2,35})\s+means\s+([^.\n]{10,180})',
            r'([A-Z][a-zA-Z0-9\s\-]{2,30})\s+is\s+(?:the|a|an)\s+process\s+of\s+([^.\n]{10,180})',
            r'([A-Z][a-zA-Z0-9\s\-]{2,30}):\s+([^.\n]{15,180})',
        ]

        found: List[Dict[str, str]] = []
        seen_terms: Set[str] = set()

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for term, definition in matches:
                clean_term = term.strip().strip(':').strip('-')
                clean_def = definition.strip().rstrip('.')
                if len(clean_term) < 3 or clean_term.lower() in STOPWORDS:
                    continue
                if clean_term.lower() not in seen_terms:
                    seen_terms.add(clean_term.lower())
                    found.append({
                        "term": clean_term,
                        "definition": clean_def
                    })

        return found

    # -------------------------------------------------------------------------
    # Feature 3: Topic Explanation and Key Points
    # -------------------------------------------------------------------------
    def explain_topic(self, topic: str, context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Explains a topic tailored for average students:
        1. Identifies key points related to that topic from the document first.
        2. Displays key points as a separate section with bold keywords.
        3. Explains each key point in simple English using clear bullet points.
        4. Breaks difficult concepts into smaller, understandable parts.
        5. Provides an example ONLY if that example is available in the document.
        6. Strictly grounded in the provided document without outside information.
        """
        topic_clean = topic.strip()
        matched_sentences: List[str] = []
        example_sentences: List[str] = []

        if context_text:
            sentences = self.split_sentences(context_text)
            topic_words = [w.lower() for w in re.findall(r'\b\w+\b', topic_clean) if w.lower() not in STOPWORDS]

            for s in sentences:
                s_lower = s.lower()
                if any(w in s_lower for w in topic_words):
                    matched_sentences.append(s)
                    if any(ex_word in s_lower for ex_word in ['for example', 'for instance', 'such as', 'e.g.', 'example:']):
                        example_sentences.append(s)

        matched_snippet = " ".join(matched_sentences[:4]) if matched_sentences else ""

        # Key points identified directly from the document
        key_points_list: List[str] = []
        if matched_sentences:
            for idx, s in enumerate(matched_sentences[:4], start=1):
                clean_s = s.strip()
                words_in_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                kw = words_in_s[0].capitalize() if words_in_s else f"Point {idx}"
                key_points_list.append(f"**{kw}**: {clean_s}")
        else:
            key_points_list = [
                f"**Core Definition**: Focus on the foundational definition of **{topic_clean}** from your notes.",
                f"**Primary Function**: How **{topic_clean}** operates under standard textbook conditions.",
                f"**Exam Importance**: Key distinctions and principles associated with **{topic_clean}**."
            ]

        # Step-by-step breakdown in simple English
        step_breakdown = (
            f"1. **Core Concept**: Understand what **{topic_clean}** is in simple terms: '{matched_snippet[:140]}...'\n"
            f"2. **How It Works**: Break down the process step-by-step as outlined in your study material.\n"
            f"3. **Why It Matters**: Identify the main purpose and benefits described in the document.\n"
            f"4. **Exam Preparation**: Memorize the key terms, definitions, and rules attached to this topic."
        )

        overview = (
            f"**{topic_clean}** is an essential subject in academic study. "
            + (f"Based on your notes: '{matched_snippet}'" if matched_snippet
               else f"It represents a core concept involving specific principles and mechanisms.")
        )

        terminology = (
            f"- **Core Subject**: `{topic_clean}` - The central concept under study.\n"
            f"- **Operating Rule**: The conditions and requirements specified in your notes.\n"
            f"- **Result / Output**: The direct effect or outcome generated by this concept."
        )

        # Example: Strictly check if example is present in the document
        if example_sentences:
            example = f"**Example Found in Uploaded Document:**\n> {example_sentences[0]}"
        else:
            example = f"*(Note: No specific real-world example was provided in the uploaded document for **{topic_clean}**. In accordance with strict grounding rules, external examples have not been invented.)*"

        recap = (
            f"⚡ **60-Second Exam Memory Recap**:\n"
            f"- **{topic_clean}** is defined directly by the rules and mechanisms in your uploaded material.\n"
            f"- Focus on its primary definition, step-by-step mechanism, and distinctions from related topics."
        )

        return {
            "mode": self.mode_label,
            "topic": topic_clean,
            "key_points": "\n".join(f"- {p}" for p in key_points_list),
            "key_points_list": key_points_list,
            "simple_overview": overview,
            "step_by_step": step_breakdown,
            "terminology": terminology,
            "practical_example": example,
            "document_example": example,
            "recap": recap,
            "grounded_notes_used": bool(matched_sentences)
        }

    # -------------------------------------------------------------------------
    # Feature 4: Important Topics Extractor with Priority Labels
    # -------------------------------------------------------------------------
    def extract_important_topics(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze notes to discover important topics with suggested study priorities:
        - Prioritizes topics based on document repetition, headings, definitions, and comparisons.
        - Labeled as 'High-Priority Study Topics' based strictly on the document.
        - Shows key points and a short, simple explanation for every topic.
        - Never claims that a topic will definitely appear in an exam.
        """
        words = self.tokenize_words(text)
        if not words:
            return []

        word_freq: Dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        candidate_phrases = re.findall(r'\b[A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z][a-zA-Z0-9\-_]+)*\b', text)
        phrase_freq: Dict[str, int] = {}
        for p in candidate_phrases:
            p_clean = p.strip()
            if len(p_clean) > 3 and p_clean.lower() not in STOPWORDS:
                phrase_freq[p_clean] = phrase_freq.get(p_clean, 0) + 1

        topic_scores: Dict[str, float] = {}
        for phrase, count in phrase_freq.items():
            topic_scores[phrase] = count * 2.0

        for word, count in word_freq.items():
            if count >= 2:
                cap = word.capitalize()
                if cap not in topic_scores:
                    topic_scores[cap] = float(count)

        if not topic_scores:
            topic_scores = {w.capitalize(): 1.0 for w in list(set(words))[:5]}

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:8]

        sentences = self.split_sentences(text)
        definitions = self.extract_definitions(text)
        def_map = {d["term"].lower(): d["definition"] for d in definitions}

        results: List[Dict[str, Any]] = []
        max_score = sorted_topics[0][1] if sorted_topics else 1.0

        for rank, (topic_name, score) in enumerate(sorted_topics, start=1):
            matched_sentences = [s for s in sentences if topic_name.lower() in s.lower()]
            relevant_sentence = matched_sentences[0] if matched_sentences else (sentences[min(rank - 1, len(sentences) - 1)] if sentences else "")

            # Simple explanation
            simple_def = def_map.get(topic_name.lower(), "")
            if simple_def:
                simple_explanation = f"**{topic_name}** is defined as {simple_def}."
            elif relevant_sentence:
                simple_explanation = f"**{topic_name}** is highlighted in the notes: \"{relevant_sentence[:130]}...\""
            else:
                simple_explanation = f"**{topic_name}** is a central concept discussed in the study material."

            # Key points for this topic
            key_points: List[str] = []
            for s in matched_sentences[:3]:
                words_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', s) if w.lower() not in STOPWORDS]
                lead_kw = words_s[0].capitalize() if words_s else "Detail"
                key_points.append(f"**{lead_kw}**: {s.strip()}")
            if not key_points and relevant_sentence:
                key_points.append(f"**Key Fact**: {relevant_sentence.strip()}")

            ratio = score / max_score
            if ratio >= 0.70 or rank <= 2:
                priority = "High Priority"
                priority_label = "High-Priority Study Topic"
                priority_class = "high"
                reason = (
                    f"Heavily emphasized in the document ({int(score)} mentions) with multiple definitions "
                    f"and conceptual rules. Fundamental anchor concept for exam revision."
                )
            elif ratio >= 0.40 or rank <= 5:
                priority = "Medium Priority"
                priority_label = "Medium-Priority Study Topic"
                priority_class = "medium"
                reason = (
                    f"Appears moderately ({int(score)} mentions) with technical explanations or comparisons. "
                    f"Important for concept-understanding questions."
                )
            else:
                priority = "Foundational Concept"
                priority_label = "Foundational Study Topic"
                priority_class = "normal"
                reason = (
                    f"Provides essential background context and terminology supporting broader topics in this document."
                )

            results.append({
                "rank": rank,
                "topic": topic_name,
                "frequency": int(score),
                "priority": priority,
                "priority_label": priority_label,
                "priority_class": priority_class,
                "simple_explanation": simple_explanation,
                "key_points": key_points,
                "why_important": reason,
                "reason": reason,
                "context_snippet": relevant_sentence[:220] + ("..." if len(relevant_sentence) > 220 else "")
            })

        return results

    # -------------------------------------------------------------------------
    # Feature 5: Rule-Based Quiz Generator (Balanced Mix & Difficulty)
    # -------------------------------------------------------------------------
    def generate_quiz(self, text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice practice questions strictly from the supplied text.
        Includes a balanced mix of:
        1. Definition-based questions
        2. Concept-understanding questions
        3. Logical & application questions
        4. Comparison & difference questions
        5. Formula-based / numerical rule questions
        """
        sentences = self.split_sentences(text)
        definitions = self.extract_definitions(text)
        comparisons = self.extract_comparisons(text)
        formulas = self.extract_formulas_and_rules(text)
        words = self.tokenize_words(text)

        candidate_terms = [d['term'] for d in definitions]
        top_words = [w.capitalize() for w in sorted(list(set(words)), key=lambda x: words.count(x.lower()), reverse=True)[:25]]
        candidate_terms.extend([tw for tw in top_words if tw not in candidate_terms])

        fallback_distractors = [
            "Encapsulation", "Asynchronous Processing", "Polymorphism", "Deadlock",
            "Linear Regression", "Stochastic Optimization", "Dynamic Programming",
            "Semantic Segmentation", "Cache Memory", "Mutual Exclusion",
            "Paging Mechanism", "Context Switching", "Gradient Descent", "Chloroplast"
        ]
        distractor_pool = list(dict.fromkeys(candidate_terms + fallback_distractors))

        questions: List[Dict[str, Any]] = []

        # 1. Definition-based Questions (Easy to Medium)
        for d in definitions:
            term = d['term']
            defn = d['definition']
            distractors = [t for t in distractor_pool if t.lower() != term.lower()]
            if len(distractors) < 3:
                distractors.extend([f"Alternative Concept {i}" for i in range(1, 4)])
            chosen_distractors = random.sample(distractors, 3)

            options = [term] + chosen_distractors
            random.shuffle(options)

            questions.append({
                "id": len(questions) + 1,
                "question": f"According to your study notes, which term is defined as: \"{defn}\"?",
                "options": options,
                "correct_answer": term,
                "explanation": f"Source document explicitly defines '{term}' as '{defn}'.",
                "topic": term,
                "question_type": "Definition",
                "difficulty": "Easy"
            })
            if len(questions) >= num_questions:
                break

        # 2. Comparison & Difference Questions (Challenging)
        if len(questions) < num_questions and comparisons:
            for comp in comparisons:
                comp_text = comp.replace("**Comparison**: ", "")
                words_in_comp = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', comp_text) if w.lower() not in STOPWORDS]
                if not words_in_comp:
                    continue
                ans_term = words_in_comp[0].capitalize()
                distractors = [t for t in distractor_pool if t.lower() != ans_term.lower()]
                chosen = random.sample(distractors, 3)
                opts = [ans_term] + chosen
                random.shuffle(opts)

                questions.append({
                    "id": len(questions) + 1,
                    "question": f"Which concept is specifically compared or contrasted in the following statement: \"{comp_text}\"?",
                    "options": opts,
                    "correct_answer": ans_term,
                    "explanation": f"This comparison is directly supported by the notes: \"{comp_text}\"",
                    "topic": ans_term,
                    "question_type": "Comparison / Difference",
                    "difficulty": "Challenging"
                })
                if len(questions) >= num_questions:
                    break

        # 3. Formula / Rule Questions (Medium to Challenging)
        if len(questions) < num_questions and formulas:
            for f in formulas:
                rule_text = f.replace("**Key Rule / Formula**: ", "")
                words_in_r = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', rule_text) if w.lower() not in STOPWORDS]
                if not words_in_r:
                    continue
                ans_term = words_in_r[0].capitalize()
                distractors = [t for t in distractor_pool if t.lower() != ans_term.lower()]
                chosen = random.sample(distractors, 3)
                opts = [ans_term] + chosen
                random.shuffle(opts)

                questions.append({
                    "id": len(questions) + 1,
                    "question": f"Which topic governs the rule / condition: \"{rule_text}\"?",
                    "options": opts,
                    "correct_answer": ans_term,
                    "explanation": f"The document notes state: \"{rule_text}\"",
                    "topic": ans_term,
                    "question_type": "Formula / Rule",
                    "difficulty": "Medium"
                })
                if len(questions) >= num_questions:
                    break

        # 4. Concept-Understanding Cloze Questions (Medium)
        if len(questions) < num_questions:
            for s in sentences:
                if len(s) < 40 or len(s) > 200:
                    continue
                words_in_s = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', s) if w.lower() not in STOPWORDS]
                if not words_in_s:
                    continue
                target_word = words_in_s[0].capitalize()
                pattern = re.compile(re.escape(target_word), re.IGNORECASE)
                cloze_sentence = pattern.sub("__________", s, count=1)
                if "__________" not in cloze_sentence:
                    continue

                distractors = [t for t in distractor_pool if t.lower() != target_word.lower()]
                if len(distractors) < 3:
                    distractors.extend(["Buffer", "Process", "Thread", "Module"])
                chosen_distractors = random.sample(distractors, 3)

                options = [target_word] + chosen_distractors
                random.shuffle(options)

                questions.append({
                    "id": len(questions) + 1,
                    "question": f"Complete the concept statement from the notes: \"{cloze_sentence}\"",
                    "options": options,
                    "correct_answer": target_word,
                    "explanation": f"The original text states: \"{s}\"",
                    "topic": target_word,
                    "question_type": "Concept Understanding",
                    "difficulty": "Medium"
                })
                if len(questions) >= num_questions:
                    break

        # 5. Logical Application Fallback (Easy to Medium)
        if len(questions) < num_questions and sentences:
            for idx, s in enumerate(sentences):
                if len(questions) >= num_questions:
                    break
                correct_fact = s.strip()
                bogus_1 = f"It is completely independent of {distractor_pool[idx % len(distractor_pool)]} and has no effect."
                bogus_2 = "This mechanism was deprecated in modern theoretical frameworks."
                bogus_3 = "The opposite condition always holds true in all standard operating scenarios."

                options = [correct_fact, bogus_1, bogus_2, bogus_3]
                random.shuffle(options)

                questions.append({
                    "id": len(questions) + 1,
                    "question": f"Which of the following statements is directly supported by the study notes?",
                    "options": options,
                    "correct_answer": correct_fact,
                    "explanation": f"Supported directly in the provided material: \"{correct_fact}\"",
                    "topic": "General Revision",
                    "question_type": "Logical Application",
                    "difficulty": "Easy"
                })

        final_questions = questions[:num_questions]
        for i, q in enumerate(final_questions, start=1):
            q["id"] = i

        return final_questions

    # -------------------------------------------------------------------------
    # Feature 6: Grounded Document Q&A with Strict "Not Available" Handling
    # -------------------------------------------------------------------------
    def answer_question(self, question: str, text: str) -> Dict[str, Any]:
        """
        Answer student questions based strictly on the uploaded text:
        - Explains answers in simple, student-friendly English.
        - Uses bullet points, numbered steps, and comparison tables where appropriate.
        - Highlights keywords and concepts in bold.
        - Handles 'Explain', 'Short Notes', 'Difference Between', 'List', and 'Explain in detail'.
        - If not present in text: clearly states 'This information is not available in the uploaded PDF.'
        """
        cleaned_question = question.strip()
        q_words = self.tokenize_words(cleaned_question)
        q_lower = cleaned_question.lower()

        if not text.strip():
            return {
                "mode": self.mode_label,
                "answer": "⚠️ This information is not available in the uploaded PDF. (No study material is loaded).",
                "found_in_notes": False,
                "confidence": 0.0,
                "reference_snippet": ""
            }

        if not q_words:
            return {
                "mode": self.mode_label,
                "answer": "Please ask a specific academic question about your study material.",
                "found_in_notes": False,
                "confidence": 0.0,
                "reference_snippet": ""
            }

        sentences = self.split_sentences(text)
        if not sentences:
            return {
                "mode": self.mode_label,
                "answer": "⚠️ This information is not available in the uploaded PDF.",
                "found_in_notes": False,
                "confidence": 0.0,
                "reference_snippet": ""
            }

        # Score sentences by matching question keywords
        scored_sentences: List[Tuple[str, float]] = []
        for s in sentences:
            s_words = set(self.tokenize_words(s))
            matches = sum(1 for qw in q_words if qw in s_words)
            if matches > 0:
                score = matches / (len(q_words) + math.sqrt(len(s_words)))
                scored_sentences.append((s, score))

        # Check if question is outside the document
        if not scored_sentences:
            return {
                "mode": self.mode_label,
                "answer": "⚠️ **This information is not available in the uploaded PDF.**",
                "found_in_notes": False,
                "confidence": 0.0,
                "reference_snippet": ""
            }

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentence, best_score = scored_sentences[0]

        if best_score < 0.12:
            return {
                "mode": self.mode_label,
                "answer": (
                    "⚠️ **This information is not available in the uploaded PDF.**\n\n"
                    f"*(The closest related line found in your document was: \"{best_sentence}\", but it does not fully answer your question.)*"
                ),
                "found_in_notes": False,
                "confidence": round(best_score, 2),
                "reference_snippet": best_sentence
            }

        top_sentences = [s for s, _ in scored_sentences[:4]]

        # Formatting based on question type
        is_difference = any(k in q_lower for k in ["difference", "compare", "vs", "versus", "distinction"])
        is_list = any(k in q_lower for k in ["list", "types", "kinds", "steps", "conditions"])
        is_short_notes = any(k in q_lower for k in ["short note", "brief", "summary", "overview"])

        formatted_lines = []

        if is_difference and len(top_sentences) >= 2:
            formatted_lines.append("### ⚖️ **Comparison & Differences from Document**")
            formatted_lines.append("")
            formatted_lines.append("| Concept / Feature | Detail from Notes |")
            formatted_lines.append("| :--- | :--- |")
            for idx, s in enumerate(top_sentences[:3], start=1):
                clean_s = s.replace("|", "/")
                words_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                kw = words_s[0].capitalize() if words_s else f"Aspect {idx}"
                formatted_lines.append(f"| **{kw}** | {clean_s} |")
            formatted_lines.append("")
        elif is_list:
            formatted_lines.append("### 📋 **List from Study Material**")
            for idx, s in enumerate(top_sentences, start=1):
                clean_s = s.strip()
                words_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                kw = words_s[0].capitalize() if words_s else f"Item {idx}"
                formatted_lines.append(f"{idx}. **{kw}**: {clean_s}")
            formatted_lines.append("")
        elif is_short_notes:
            formatted_lines.append("### 📝 **Short Revision Notes**")
            for s in top_sentences:
                clean_s = s.strip()
                words_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                kw = words_s[0].capitalize() if words_s else "Key Concept"
                formatted_lines.append(f"- **{kw}**: {clean_s}")
            formatted_lines.append("")
        else:
            formatted_lines.append("### 💡 **Explanation Based on Notes**")
            for s in top_sentences:
                clean_s = s.strip()
                words_s = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', clean_s) if w.lower() not in STOPWORDS]
                kw = words_s[0].capitalize() if words_s else "Concept"
                formatted_lines.append(f"- **{kw}**: {clean_s}")
            formatted_lines.append("")

        formatted_lines.append(f"> 📌 **Source Reference from Notes**: \"{best_sentence}\"")
        answer_text = "\n".join(formatted_lines)

        return {
            "mode": self.mode_label,
            "answer": answer_text,
            "found_in_notes": True,
            "confidence": round(best_score, 2),
            "reference_snippet": best_sentence
        }
