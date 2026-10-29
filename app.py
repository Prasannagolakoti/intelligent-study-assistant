"""
Intelligent Study Assistant Agent - Main Application
A modern, responsive educational dashboard tailored for average college students.
Provides:
1. Extracted Notes Generator (simplified student notes topic-by-topic)
2. AI Study Summaries (clear bullet points with bold keywords, definitions & formulas)
3. Topic Explanation & Key Points (separate key points section, simple English)
4. High-Priority Study Topics Extractor (key points, simple explanations & emphasis reasons)
5. Interactive Practice Quizzes (balanced question types, difficulty levels & explanations)
6. Grounded Document Q&A (strict PDF grounding, comparison tables, not-found notices)
7. Study History & Activity Log
"""

import os
import streamlit as st
from dotenv import load_dotenv

from modules.document_processor import DocumentProcessor, DocumentProcessingError
from modules.study_assistant import StudyAssistant
from modules.quiz_generator import QuizGenerator
from modules.topic_extractor import TopicExtractor
from modules.history_manager import HistoryManager

# Page Configuration
st.set_page_config(
    page_title="Intelligent Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Styles
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Core Services
processor = DocumentProcessor()
assistant = StudyAssistant()
quiz_generator = QuizGenerator()
topic_extractor = TopicExtractor()
history_manager = HistoryManager()

# Initialize Session State
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_title" not in st.session_state:
    st.session_state.document_title = "No Document Loaded"
if "document_metrics" not in st.session_state:
    st.session_state.document_metrics = {}
if "extracted_notes_result" not in st.session_state:
    st.session_state.extracted_notes_result = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "topics_result" not in st.session_state:
    st.session_state.topics_result = None
if "selected_topic_query" not in st.session_state:
    st.session_state.selected_topic_query = ""
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "user_quiz_answers" not in st.session_state:
    st.session_state.user_quiz_answers = {}
if "quiz_score_result" not in st.session_state:
    st.session_state.quiz_score_result = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "custom_api_key" not in st.session_state:
    st.session_state.custom_api_key = ""
if "force_local_mode" not in st.session_state:
    st.session_state.force_local_mode = False


# -----------------------------------------------------------------------------
# SIDEBAR: Navigation & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Intelligent Study Assistant")
    st.caption("AI-Powered College Exam Preparation for Average Students")
    st.markdown("---")

    # Navigation Menu
    PAGES = [
        "🏠 Home Dashboard",
        "📄 Upload Study Material",
        "📖 Extracted Notes",
        "📝 AI Study Summary",
        "💡 Explain a Topic",
        "🎯 Important Topics",
        "❓ Quiz Generator",
        "📊 Quiz Results",
        "💬 Ask Notes (Q&A)",
        "📜 Study History",
        "ℹ️ About Project"
    ]
    selected_page = st.radio("Navigation Menu", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### ⚙️ AI Engine Settings")

    # API Configuration
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.custom_api_key,
        type="password",
        help="Enter your free Google Gemini API key from https://aistudio.google.com/ or leave blank for Local Demonstration Mode."
    )
    if api_key_input != st.session_state.custom_api_key:
        st.session_state.custom_api_key = api_key_input
        st.rerun()

    force_local = st.toggle("Force Local Demo Mode", value=st.session_state.force_local_mode)
    if force_local != st.session_state.force_local_mode:
        st.session_state.force_local_mode = force_local
        st.rerun()

    # Active Provider Status
    effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
    provider_info = assistant.get_active_provider_info(effective_key)

    st.markdown(f"**Status:** {provider_info['status_badge']}")
    st.caption(provider_info["description"])

    st.markdown("---")
    st.markdown("#### 📌 Current Study Material")
    if st.session_state.document_text:
        words = st.session_state.document_metrics.get("word_count", 0)
        st.success(f"**{st.session_state.document_title}**\n\n({words:,} words loaded)")
    else:
        st.info("No study material loaded yet. Visit 'Upload Study Material' or choose a sample.")


# -----------------------------------------------------------------------------
# HELPER: Load Sample Material
# -----------------------------------------------------------------------------
def load_sample(sample_name: str):
    path = os.path.join(os.path.dirname(__file__), "sample_data", f"{sample_name}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        title = "Operating Systems (Deadlocks & Sync)" if sample_name == "operating_systems" else "Machine Learning (Supervised Learning)"
        st.session_state.document_text = processor.clean_text(content)
        st.session_state.document_title = title
        st.session_state.document_metrics = processor.get_document_metrics(content)
        # Reset downstream results
        st.session_state.extracted_notes_result = None
        st.session_state.summary_result = None
        st.session_state.topics_result = None
        st.session_state.selected_topic_query = ""
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score_result = None
        st.session_state.chat_messages = []


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if selected_page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="hero-header">
        <h1>Intelligent Study Assistant</h1>
        <p>Your student-friendly academic partner. Converts complex textbook PDFs into simple notes, clear bullet-point summaries, high-priority study topics, and exam-practice quizzes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Metric Stats
    stats = history_manager.get_summary_stats()
    current_doc_words = st.session_state.document_metrics.get("word_count", 0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{current_doc_words:,}</div>
            <div class="stat-label">Words in Current Notes</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{stats['total_summaries']}</div>
            <div class="stat-label">Summaries Generated</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{stats['total_quizzes']}</div>
            <div class="stat-label">Quizzes Completed</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{stats['avg_quiz_score']}%</div>
            <div class="stat-label">Avg Quiz Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Study Actions for Average Students")
    st.caption("Jump straight into simplified study features or load a ready-to-test college syllabus dataset.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="study-card">
            <div class="card-title">📖 1. Extracted Notes</div>
            <div class="card-desc">Converts complex textbook chapters into simple, student-friendly notes with plain English, clear headings, and bold keywords.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Extracted Notes", key="btn_home_notes"):
            st.info("Please select '📖 Extracted Notes' in the sidebar.")

    with c2:
        st.markdown("""
        <div class="study-card">
            <div class="card-title">📝 2. AI Study Summary</div>
            <div class="card-desc">Clear exam revision bullets highlighting definitions, formulas, and comparisons in bold for quick memorization.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Summary", key="btn_home_sum"):
            st.info("Please select '📝 AI Study Summary' in the sidebar.")

    with c3:
        st.markdown("""
        <div class="study-card">
            <div class="card-title">❓ 3. Practice Quiz</div>
            <div class="card-desc">Balanced mix of definition, concept understanding, comparison, and application MCQs with instant explanations.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Quiz", key="btn_home_quiz"):
            st.info("Please select '❓ Quiz Generator' in the sidebar.")

    st.markdown("---")
    st.markdown("### 🧪 Instant Demonstration Presets")
    st.caption("Don't have your PDF notes ready? Load one of our curated college study datasets with one click:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📚 Load Sample: Operating Systems (Process Synchronization & Deadlocks)", use_container_width=True):
            load_sample("operating_systems")
            st.success("Loaded Operating Systems sample notes! Select any feature in the sidebar to begin.")
            st.rerun()

    with col_s2:
        if st.button("🤖 Load Sample: Machine Learning (Supervised Learning & Regression)", use_container_width=True):
            load_sample("machine_learning")
            st.success("Loaded Machine Learning sample notes! Select any feature in the sidebar to begin.")
            st.rerun()


# =============================================================================
# PAGE 2: UPLOAD STUDY MATERIAL
# =============================================================================
elif selected_page == "📄 Upload Study Material":
    st.markdown('<div class="page-title">📄 Upload & Process Study Material</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload lecture notes, textbooks, or paste text to analyze. Supports searchable PDF and TXT.</div>', unsafe_allow_html=True)

    tab_file, tab_paste, tab_samples = st.tabs(["📁 File Upload (PDF / TXT)", "✍️ Paste Study Notes", "📚 Sample Library"])

    with tab_file:
        uploaded_file = st.file_uploader("Choose a study file", type=["pdf", "txt"], help="Limit 20MB per file.")

        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            file_size = len(file_bytes)

            is_valid, err_msg = processor.validate_file(file_name, file_size)
            if not is_valid:
                st.error(f"❌ Validation Error: {err_msg}")
            else:
                st.success(f"✅ File '{file_name}' ({file_size/1024:.1f} KB) passed integrity validation.")

                if st.button("Process & Extract Study Material", type="primary"):
                    with st.spinner("Extracting text and analyzing document structure..."):
                        try:
                            if file_name.lower().endswith(".pdf"):
                                raw_text, pages = processor.extract_from_pdf(file_bytes)
                            else:
                                raw_text = processor.extract_from_txt(file_bytes)
                                pages = 1

                            cleaned = processor.clean_text(raw_text)
                            metrics = processor.get_document_metrics(cleaned, page_count=pages)

                            st.session_state.document_text = cleaned
                            st.session_state.document_title = file_name
                            st.session_state.document_metrics = metrics
                            st.session_state.extracted_notes_result = None
                            st.session_state.summary_result = None
                            st.session_state.topics_result = None
                            st.session_state.quiz_data = None
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_score_result = None

                            st.success(f"Successfully processed **{file_name}**! Choose '📖 Extracted Notes' or '📝 AI Study Summary' next.")
                            st.rerun()

                        except DocumentProcessingError as ex:
                            st.error(f"⚠️ Document Extraction Notice: {str(ex)}")

    with tab_paste:
        custom_title = st.text_input("Study Notes Title", value="My Custom Study Notes")
        pasted_content = st.text_area("Paste Notes / Textbook Content Here", height=250, placeholder="Paste your study paragraphs or lecture transcript here...")

        if st.button("Save & Process Pasted Notes", type="primary"):
            if not pasted_content.strip():
                st.warning("Please paste some content before saving.")
            else:
                cleaned = processor.clean_text(pasted_content)
                metrics = processor.get_document_metrics(cleaned)
                st.session_state.document_text = cleaned
                st.session_state.document_title = custom_title.strip() or "Custom Notes"
                st.session_state.document_metrics = metrics
                st.session_state.extracted_notes_result = None
                st.session_state.summary_result = None
                st.session_state.topics_result = None
                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score_result = None
                st.success("Pasted material saved and analyzed successfully!")
                st.rerun()

    with tab_samples:
        st.markdown("#### Ready-to-Test Academic Materials")
        st.write("Click below to test the pipeline immediately with comprehensive college notes:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📖 Load Operating Systems Notes", use_container_width=True):
                load_sample("operating_systems")
                st.success("Loaded Operating Systems sample notes!")
                st.rerun()
        with c2:
            if st.button("🤖 Load Machine Learning Notes", use_container_width=True):
                load_sample("machine_learning")
                st.success("Loaded Machine Learning sample notes!")
                st.rerun()

    # Active Document Inspection Preview
    if st.session_state.document_text:
        st.markdown("---")
        st.markdown("### 📊 Document Overview & Metrics")

        m = st.session_state.document_metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Words", f"{m.get('word_count', 0):,}")
        c2.metric("Characters", f"{m.get('char_count', 0):,}")
        c3.metric("Sentences", f"{m.get('sentence_count', 0):,}")
        c4.metric("Est. Reading Time", f"{m.get('estimated_reading_minutes', 0)} min")
        c5.metric("Pages", f"{m.get('page_count', 1)}")

        with st.expander("🔍 View Extracted Notes Preview (First 2,500 characters)", expanded=False):
            st.text(st.session_state.document_text[:2500] + ("..." if len(st.session_state.document_text) > 2500 else ""))


# =============================================================================
# PAGE 3: EXTRACTED NOTES GENERATOR (NEW FEATURE 1)
# =============================================================================
elif selected_page == "📖 Extracted Notes":
    st.markdown('<div class="page-title">📖 Extracted Notes Generator (Simple English)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Extract and present the complete relevant study content from your uploaded document in simple, easy-to-understand English. Complex textbook paragraphs are simplified with clear headings, subheadings, and bold keywords.</div>', unsafe_allow_html=True)

    if not st.session_state.document_text:
        st.warning("⚠️ No study material is currently loaded. Please upload a PDF in '📄 Upload Study Material' or choose a sample from the library.")
    else:
        st.info(f"Target Material: **{st.session_state.document_title}** ({st.session_state.document_metrics.get('word_count', 0)} words)")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("Generate comprehensive, student-friendly notes organized topic-by-topic with difficult terms explained simply.")
        with c2:
            gen_notes_btn = st.button("⚡ Generate Extracted Notes Now", type="primary", use_container_width=True)

        if gen_notes_btn or (st.session_state.extracted_notes_result is None and st.session_state.document_text):
            with st.spinner("Simplifying textbook content, organizing topics, and building glossary..."):
                effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                notes_res = assistant.generate_extracted_notes(
                    st.session_state.document_text,
                    custom_key=effective_key
                )
                st.session_state.extracted_notes_result = notes_res

        if st.session_state.extracted_notes_result:
            res = st.session_state.extracted_notes_result

            # Mode Attribution Badge
            if res.get("is_llm"):
                st.markdown('<span class="badge-status badge-gemini">🟢 Generated via Google Gemini LLM (Simplified for Students)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-status badge-local">🟡 Generated via Local Demonstration Mode (Extractive NLP)</span>', unsafe_allow_html=True)

            if "warning" in res:
                st.warning(res["warning"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics
            topics_covered = res.get("topics_covered", [])
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Topics Covered in Notes", len(topics_covered))
            col_m2.metric("Extracted Word Count", f"{res.get('word_count', 0):,}")

            st.markdown("---")
            # Render Markdown Notes
            st.markdown(res.get("notes_markdown", ""))

            st.markdown("---")
            # Download Extracted Notes
            notes_md_text = res.get("notes_markdown", "")
            st.download_button(
                label="📥 Download Extracted Notes as Markdown (.md)",
                data=notes_md_text,
                file_name=f"Simplified_Notes_{st.session_state.document_title.replace(' ', '_')}.md",
                mime="text/markdown"
            )


# =============================================================================
# PAGE 4: AI STUDY SUMMARY (FEATURE 2)
# =============================================================================
elif selected_page == "📝 AI Study Summary":
    st.markdown('<div class="page-title">📝 AI Study Summary Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Exam-oriented summaries presented in clear bullet points instead of long paragraphs. Important keywords, definitions, formulas, and comparisons are highlighted in bold.</div>', unsafe_allow_html=True)

    if not st.session_state.document_text:
        st.warning("⚠️ No study material is currently loaded. Please upload notes or choose a sample in '📄 Upload Study Material'.")
    else:
        st.info(f"Target Material: **{st.session_state.document_title}** ({st.session_state.document_metrics.get('word_count', 0)} words)")

        c1, c2 = st.columns([2, 1])
        with c1:
            length_choice = st.select_slider(
                "Select Desired Summary Depth:",
                options=["Short", "Medium", "Detailed"],
                value="Medium",
                help="Short: Rapid high-level review. Medium: Balanced exam review. Detailed: In-depth revision breakdown."
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_btn = st.button("✨ Generate Study Summary", type="primary", use_container_width=True)

        if gen_btn:
            with st.spinner("Analyzing text, ranking sentences, and synthesizing summary in clear bullets..."):
                effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                res = assistant.generate_summary(
                    st.session_state.document_text,
                    length=length_choice,
                    custom_key=effective_key
                )
                st.session_state.summary_result = res

                # Record in history
                history_manager.add_summary_record(
                    document_title=st.session_state.document_title,
                    summary_type=length_choice,
                    executive_summary=res.get("executive_summary", ""),
                    mode=res.get("mode", "Unknown"),
                    word_count=st.session_state.document_metrics.get("word_count", 0)
                )

        if st.session_state.summary_result:
            res = st.session_state.summary_result

            # Mode Attribution Badge
            if res.get("is_llm"):
                st.markdown('<span class="badge-status badge-gemini">🟢 Generated via Google Gemini LLM</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-status badge-local">🟡 Generated via Local Demonstration Mode (Extractive NLP)</span>', unsafe_allow_html=True)

            if "warning" in res:
                st.warning(res["warning"])

            st.markdown("<br>", unsafe_allow_html=True)

            # 1. Executive Summary in Clear Bullet Points (No long paragraphs!)
            st.markdown(f"### 📌 Executive Revision Overview ({res.get('summary_type', 'Medium')} Review)")
            exec_bullets = res.get("executive_summary_bullets", [])
            if exec_bullets:
                for b in exec_bullets:
                    st.markdown(f"- {b}")
            else:
                st.write(res.get("executive_summary", ""))

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Core Concepts
            concepts = res.get("core_concepts", [])
            if concepts:
                st.markdown("### 🔑 Core Concepts & Key Ideas")
                pills_html = " ".join(f'<span style="background:#eff6ff; color:#000000; padding:4px 10px; border-radius:6px; font-weight:700; margin-right:6px; font-size:0.88rem; border:1px solid #cbd5e1;">{c}</span>' for c in concepts)
                st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Key Definitions
            definitions = res.get("key_definitions", [])
            st.markdown("### 📖 Essential Definitions (Explained Simply)")
            if definitions:
                for d in definitions:
                    with st.container():
                        st.markdown(f"- **{d.get('term', '')}**: {d.get('definition', '')}")
            else:
                st.caption("No explicit formal definitions found in this segment.")

            # 4. Comparisons & Differences
            comparisons = res.get("comparisons", [])
            if comparisons:
                st.markdown("### ⚖️ Comparisons, Differences & Advantages / Disadvantages")
                for comp in comparisons:
                    st.markdown(f"- {comp}")

            # 5. Formulas & Key Rules
            formulas = res.get("formulas_and_rules", [])
            if formulas:
                st.markdown("### 🧮 Formulas, Rules & Key Statements")
                for rule in formulas:
                    st.markdown(f"- {rule}")

            # 6. Salient Exam Revision Points
            st.markdown("### 📝 Must-Memorize Exam Points")
            bullets = res.get("bullet_points", [])
            for bp in bullets:
                st.markdown(f"- {bp}")

            st.markdown("---")
            # Download summary
            summary_txt = res.get("full_markdown", "")
            st.download_button(
                label="📥 Download Summary as Markdown (.md)",
                data=summary_txt,
                file_name=f"Study_Summary_{st.session_state.document_title.replace(' ', '_')}.md",
                mime="text/markdown"
            )


# =============================================================================
# PAGE 5: TOPIC EXPLANATION & KEY POINTS (FEATURE 3)
# =============================================================================
elif selected_page == "💡 Explain a Topic":
    st.markdown('<div class="page-title">💡 Topic Explanation & Key Points</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Demystify challenging topics: Key points identified first from your document, followed by simple step-by-step explanations, bold keywords, and document-backed examples.</div>', unsafe_allow_html=True)

    default_topic = st.session_state.selected_topic_query or ""

    topic_query = st.text_input(
        "Enter a topic, mechanism, or academic question:",
        value=default_topic,
        placeholder="e.g. Critical Section Problem, Banker's Algorithm, Bias-Variance Tradeoff, Gradient Descent",
        help="Type any concept from your notes you wish to understand in simple terms."
    )

    use_doc_context = st.checkbox("Ground explanation using loaded study notes", value=bool(st.session_state.document_text))

    if st.button("🚀 Explain This Concept", type="primary"):
        if not topic_query.strip():
            st.warning("Please type a topic to explain.")
        else:
            with st.spinner(f"Identifying key points and preparing beginner-friendly explanation for '{topic_query}'..."):
                effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                context_to_use = st.session_state.document_text if use_doc_context else None

                exp = assistant.explain_topic(
                    topic=topic_query,
                    context_text=context_to_use,
                    custom_key=effective_key
                )
                st.session_state["last_explanation"] = exp

    if "last_explanation" in st.session_state:
        exp = st.session_state["last_explanation"]

        if exp.get("is_llm"):
            st.markdown('<span class="badge-status badge-gemini">🟢 Explained with Google Gemini LLM</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-status badge-local">🟡 Explained with Local Demonstration Engine</span>', unsafe_allow_html=True)

        if exp.get("grounded_notes_used"):
            st.caption(f"✓ Grounded strictly in: '{st.session_state.document_title}'")

        st.markdown(f"## 📘 Understanding: **{exp.get('topic', topic_query)}**")

        # REQUIREMENT 3: Separate Key Points section displayed FIRST
        st.markdown("### 🎯 1. Key Points Identified from Notes")
        key_pts = exp.get("key_points_list", [])
        if key_pts:
            for p in key_pts:
                st.markdown(f"- {p}")
        else:
            st.markdown(exp.get("key_points", "- Key points extracted from document."))

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview
        st.markdown("### 🌟 2. Simple English Overview")
        st.info(exp.get("simple_overview", ""))

        # Step-by-step breakdown
        st.markdown("### 🪜 3. Step-by-Step Breakdown (Broken into Simple Parts)")
        st.write(exp.get("step_by_step", ""))

        # Terminology
        st.markdown("### 📖 4. Essential Terminology & Definitions")
        st.markdown(exp.get("terminology", ""))

        # Example (STRICTLY from document)
        st.markdown("### 🔬 5. Example from Document")
        doc_ex = exp.get("document_example") or exp.get("practical_example", "")
        st.success(doc_ex)

        # 60-Second Recap
        st.markdown("### ⚡ 6. Quick 60-Second Exam Memory Recap")
        st.warning(exp.get("recap", ""))


# =============================================================================
# PAGE 6: HIGH-PRIORITY STUDY TOPICS (FEATURE 4)
# =============================================================================
elif selected_page == "🎯 Important Topics":
    st.markdown('<div class="page-title">🎯 High-Priority Study Topics Extractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Identifies only the most important topics based on textual emphasis, headings, definitions, and repetitions in your uploaded document. Shows key points and simple explanations for each topic.</div>', unsafe_allow_html=True)

    st.markdown("""
    > 📌 **Exam Preparation Notice**: *These topics are identified as **High-Priority Study Topics** based on textual frequency, definition density, and structural cues in your uploaded document. They are intended for focused revision and do not guarantee specific examination questions.*
    """)

    if not st.session_state.document_text:
        st.warning("⚠️ No study material loaded. Please upload notes or choose a sample in '📄 Upload Study Material'.")
    else:
        st.info(f"Target Material: **{st.session_state.document_title}**")

        if st.button("🔍 Extract High-Priority Study Topics", type="primary"):
            with st.spinner("Analyzing document vocabulary, co-occurrences, and structural weight..."):
                effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                topics = topic_extractor.extract_topics(
                    st.session_state.document_text,
                    custom_key=effective_key
                )
                st.session_state.topics_result = topics

                # Record in history
                history_manager.add_topics_record(
                    document_title=st.session_state.document_title,
                    topics=topics,
                    mode="Google Gemini" if effective_key else "Local Demonstration Mode"
                )

        if st.session_state.topics_result:
            topics = st.session_state.topics_result

            st.markdown(f"### 🎯 Identified High-Priority Study Topics ({len(topics)} Topics)")

            for t in topics:
                priority = t.get("priority", "Medium Priority")
                p_label = t.get("priority_label", f"{priority} Study Topic")
                p_class = "priority-high" if "High" in priority else ("priority-medium" if "Medium" in priority else "priority-normal")

                with st.container():
                    st.markdown(f"""
                    <div class="study-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="font-size:1.2rem; font-weight:800; color:#000000;">#{t.get('rank', 1)} {t.get('topic', '')}</span>
                            <span class="priority-tag {p_class}">⭐ {p_label}</span>
                        </div>
                        <div class="card-desc" style="font-size:0.95rem; margin-bottom:0.6rem;">
                            <strong>📌 Simple Explanation:</strong> {t.get('simple_explanation', t.get('reason', ''))}
                        </div>
                    """, unsafe_allow_html=True)

                    key_pts = t.get("key_points", [])
                    if key_pts:
                        st.markdown("**🔑 Key Points to Revise:**")
                        for p in key_pts:
                            st.markdown(f"- {p}")

                    st.markdown(f"""
                        <div style="font-size:0.85rem; color:#000000; background:#f8fafc; padding:0.5rem 0.7rem; border-radius:6px; margin-top:0.5rem; border:1px solid #e2e8f0;">
                            <strong>📊 Document Emphasis Reason:</strong> {t.get('why_important', t.get('reason', ''))}<br>
                            <em>Context: "{t.get('context_snippet', '')}"</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"💡 Explain '{t.get('topic', '')}' in Detail", key=f"btn_exp_topic_{t.get('rank', 1)}"):
                        st.session_state.selected_topic_query = t.get('topic', '')
                        st.info(f"Topic '{t.get('topic', '')}' selected! Please click '💡 Explain a Topic' in the sidebar.")


# =============================================================================
# PAGE 7: QUIZ GENERATOR (FEATURE 5)
# =============================================================================
elif selected_page == "❓ Quiz Generator":
    st.markdown('<div class="page-title">❓ Interactive Practice Quiz Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate meaningful multiple-choice practice quizzes from your study material. Includes definition, concept understanding, logical application, comparison, and formula questions.</div>', unsafe_allow_html=True)

    st.markdown("""
    > 📌 **Exam Practice Notice**: *Questions are generated from textual emphasis in your uploaded document for revision and self-assessment. They do not guarantee appearance on actual university exams.*
    """)

    if not st.session_state.document_text:
        st.warning("⚠️ No study material loaded. Please upload notes or choose a sample in '📄 Upload Study Material'.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            q_count = st.selectbox("Number of Questions", [5, 10, 15], index=0)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_quiz_btn = st.button("🎲 Generate Quiz Now", type="primary", use_container_width=True)

        if gen_quiz_btn:
            with st.spinner(f"Generating {q_count} balanced practice questions with distractors and explanations..."):
                effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                quiz_data = quiz_generator.generate_quiz(
                    st.session_state.document_text,
                    num_questions=q_count,
                    custom_key=effective_key
                )
                st.session_state.quiz_data = quiz_data
                st.session_state.user_quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score_result = None
                st.success(f"Generated {quiz_data['question_count']} exam-practice questions!")

        # Interactive Quiz Interface
        if st.session_state.quiz_data and st.session_state.quiz_data.get("questions"):
            q_data = st.session_state.quiz_data
            questions = q_data["questions"]

            if q_data.get("is_llm"):
                st.markdown('<span class="badge-status badge-gemini">🟢 Generated via Google Gemini LLM</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-status badge-local">🟡 Generated via Local Rule-Based Engine</span>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### 📝 Practice Exam: {st.session_state.document_title} ({len(questions)} Questions)")
            st.caption("Answer keys and explanations remain hidden until you submit your quiz.")

            with st.form("interactive_quiz_form"):
                for idx, q in enumerate(questions, start=1):
                    q_type = q.get("question_type", "Concept Understanding")
                    diff = q.get("difficulty", "Medium")
                    diff_class = f"diff-badge-{diff.lower()}"

                    st.markdown(f"""
                    <div class="quiz-question-box">
                        <div style="margin-bottom:0.5rem;">
                            <span class="quiz-badge">Question {idx} of {len(questions)} • {q.get('topic', 'General')}</span>
                            <span class="type-badge">🏷️ {q_type}</span>
                            <span class="{diff_class}">⚡ {diff}</span>
                        </div>
                        <div style="font-weight:700; font-size:1.05rem; margin-bottom:0.75rem; color:#000000;">{q['question']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    current_val = st.session_state.user_quiz_answers.get(q["id"], None)
                    options = q["options"]

                    chosen = st.radio(
                        f"Select your answer for Question {idx}:",
                        options=options,
                        index=None if current_val is None else options.index(current_val) if current_val in options else None,
                        key=f"q_radio_{q['id']}"
                    )
                    st.session_state.user_quiz_answers[q["id"]] = chosen
                    st.markdown("<br>", unsafe_allow_html=True)

                submit_btn = st.form_submit_button("🏁 Submit Quiz for Scoring", type="primary")

                if submit_btn:
                    score_res = quiz_generator.score_quiz(questions, st.session_state.user_quiz_answers)
                    st.session_state.quiz_score_result = score_res
                    st.session_state.quiz_submitted = True

                    history_manager.add_quiz_record(
                        document_title=st.session_state.document_title,
                        score_data=score_res,
                        mode=q_data.get("mode", "Local")
                    )

                    st.success("Quiz submitted successfully! Viewing results...")
                    st.rerun()


# =============================================================================
# PAGE 8: QUIZ RESULTS & ANALYTICS
# =============================================================================
elif selected_page == "📊 Quiz Results":
    st.markdown('<div class="page-title">📊 Quiz Results & Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review your score, correct answers, explanations, and personalized revision recommendations.</div>', unsafe_allow_html=True)

    if not st.session_state.quiz_submitted or not st.session_state.quiz_score_result:
        st.warning("⚠️ No quiz has been submitted yet in this session. Visit '❓ Quiz Generator' to take a quiz.")
    else:
        res = st.session_state.quiz_score_result
        total = res["total_questions"]
        correct = res["correct_count"]
        pct = res["percentage"]
        band = res["performance_band"]

        # Score Banner Card
        st.markdown(f"""
        <div class="hero-header" style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #7c3aed 100%);">
            <h1 style="font-size:2.4rem; color:#000000 !important;">{correct} / {total} Correct ({pct}%)</h1>
            <p style="font-size:1.15rem; font-weight:600; color:#000000 !important;">{band}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score Percentage", f"{pct}%")
        m2.metric("Correct Answers", f"{correct}")
        m3.metric("Incorrect / Skipped", f"{total - correct}")
        m4.metric("Questions Evaluated", f"{total}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Revision Recommendations Box
        st.markdown("### 🎯 Revision Topic Recommendations")
        if res.get("weak_topics"):
            st.warning(f"**Topics Requiring Additional Study:**\n\n{res['revision_advice']}")
        else:
            st.success(f"🎉 **Mastery Achieved:** {res['revision_advice']}")

        st.markdown("---")
        st.markdown("### 🔍 Question-by-Question Detailed Review")

        for r in res.get("detailed_results", []):
            is_c = r["is_correct"]
            icon = "✅" if is_c else "❌"
            badge_bg = "#dcfce7" if is_c else "#fee2e2"
            badge_color = "#000000"

            st.markdown(f"""
            <div class="study-card" style="border-left: 5px solid {'#10b981' if is_c else '#ef4444'};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <span style="font-weight:700; font-size:1rem; color:#000000;">Question {r['id']}: {r['question']}</span>
                    <span style="background:{badge_bg}; color:{badge_color}; font-weight:700; padding:3px 8px; border-radius:6px; font-size:0.8rem;">
                        {icon} {'CORRECT' if is_c else 'INCORRECT'}
                    </span>
                </div>
                <div style="font-size:0.92rem; margin-bottom:0.35rem; color:#000000;">
                    <strong>Your Choice:</strong> <span style="color:#000000; font-weight:600;">{r['selected_answer']}</span>
                </div>
                <div style="font-size:0.92rem; margin-bottom:0.5rem; color:#000000;">
                    <strong>Correct Answer:</strong> <span style="color:#000000; font-weight:700;">{r['correct_answer']}</span>
                </div>
                <div class="quiz-explanation-box">
                    <strong>💡 Simple Explanation:</strong> {r['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔄 Retake or Generate New Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score_result = None
            st.session_state.user_quiz_answers = {}
            st.info("Quiz state cleared. Please select '❓ Quiz Generator' in the sidebar.")


# =============================================================================
# PAGE 9: ASK NOTES (FEATURE 6: GROUNDED Q&A)
# =============================================================================
elif selected_page == "💬 Ask Notes (Q&A)":
    st.markdown('<div class="page-title">💬 Ask Questions About Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Chat-style assistant grounded strictly in your uploaded PDF. Answers are explained in simple English with bullet points, numbered steps, comparison tables, and bold keywords.</div>', unsafe_allow_html=True)

    st.markdown("""
    > ⚠️ **Strict Document Grounding Rule**: *The assistant uses only your uploaded document. If the requested information is not present in the document, it will explicitly say: **"This information is not available in the uploaded PDF."** rather than guessing.*
    """)

    if not st.session_state.document_text:
        st.warning("⚠️ No study material loaded. Please upload notes or choose a sample in '📄 Upload Study Material'.")
    else:
        st.info(f"Connected Document: **{st.session_state.document_title}** ({st.session_state.document_metrics.get('word_count', 0)} words)")

        # Display Chat History
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question about your study material (e.g. Explain, Short Notes, Difference between...)...")

        if user_query:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document context and verifying references..."):
                    effective_key = None if st.session_state.force_local_mode else st.session_state.custom_api_key
                    qa_result = assistant.answer_question(
                        question=user_query,
                        text=st.session_state.document_text,
                        chat_history=st.session_state.chat_messages[:-1],
                        custom_key=effective_key
                    )
                    reply_text = qa_result["answer"]
                    st.markdown(reply_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat Conversation"):
                st.session_state.chat_messages = []
                st.rerun()


# =============================================================================
# PAGE 10: STUDY HISTORY
# =============================================================================
elif selected_page == "📜 Study History":
    st.markdown('<div class="page-title">📜 Study History & Activity Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review previous summaries, quiz scores, and topic analyses. Stored securely on your local machine.</div>', unsafe_allow_html=True)

    records = history_manager.get_history()
    quizzes = records.get("quizzes", [])
    summaries = records.get("summaries", [])
    topics = records.get("topics", [])

    tab_q, tab_s, tab_t = st.tabs([
        f"📝 Quiz Attempts ({len(quizzes)})",
        f"📋 Generated Summaries ({len(summaries)})",
        f"🎯 Topic Analyses ({len(topics)})"
    ])

    with tab_q:
        if not quizzes:
            st.caption("No quiz attempts recorded yet.")
        else:
            for q in quizzes:
                st.markdown(f"""
                <div class="study-card">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>{q['document_title']}</strong>
                        <span style="color:#000000; font-size:0.85rem;">{q['timestamp']}</span>
                    </div>
                    <div style="font-size:1.1rem; font-weight:700; color:#000000; margin:0.3rem 0;">
                        Score: {q['correct_count']} / {q['total_questions']} ({q['percentage']}%) • {q['performance_band']}
                    </div>
                    <div style="font-size:0.85rem; color:#000000;">
                        Engine: {q['mode']} | Missed Topics: {', '.join(q.get('weak_topics', [])) or 'None (100% Score)'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_s:
        if not summaries:
            st.caption("No summaries recorded yet.")
        else:
            for s in summaries:
                st.markdown(f"""
                <div class="study-card">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>{s['document_title']} ({s['summary_type']})</strong>
                        <span style="color:#000000; font-size:0.85rem;">{s['timestamp']}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#000000; margin-top:0.4rem;">
                        {s['executive_summary']}
                    </div>
                    <div style="font-size:0.8rem; color:#000000; margin-top:0.3rem;">
                        Engine: {s['mode']} • Document Words: {s['word_count']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_t:
        if not topics:
            st.caption("No topic extractions recorded yet.")
        else:
            for t in topics:
                st.markdown(f"""
                <div class="study-card">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>{t['document_title']}</strong>
                        <span style="color:#000000; font-size:0.85rem;">{t['timestamp']}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#000000; margin-top:0.3rem;">
                        Top Concepts: {', '.join(t.get('top_topics', []))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    export_json = history_manager.export_as_json()
    st.download_button(
        "📥 Export Study History (JSON)",
        data=export_json,
        file_name="study_history_export.json",
        mime="application/json"
    )

    if st.button("🗑️ Clear Entire History", type="secondary"):
        history_manager.clear_history()
        st.success("Study history cleared successfully!")
        st.rerun()


# =============================================================================
# PAGE 11: ABOUT PROJECT
# =============================================================================
elif selected_page == "ℹ️ About Project":
    st.markdown('<div class="page-title">ℹ️ About Intelligent Study Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">College Mini-Capstone Project in Applied Artificial Intelligence and Natural Language Processing.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    **Intelligent Study Assistant** is an agentic academic companion engineered to make learning easy for average students by providing simple, clear, exam-oriented study materials based strictly on uploaded documents.

    ### 🌟 Core Student-Friendly Capabilities
    1. **📖 Extracted Notes Generator**: Rewrites complete PDF study material into simple, easy-to-understand English with clear headings, subheadings, bullet points, and difficult words explained in simple terms.
    2. **📝 AI Study Summary Generator**: Summaries in clear bullet points highlighting definitions, formulas, and comparisons in bold for easy memorization.
    3. **💡 Topic Explanation & Key Points**: Identifies key points first, breaks complex ideas into smaller parts, and cites examples only from the document.
    4. **🎯 High-Priority Study Topics**: Discovers the most important topics based on textual emphasis, repeated concepts, and definitions.
    5. **❓ Balanced Practice Quizzes**: Balanced mix of definition, concept understanding, comparison, logical application, and formula questions with instant explanations.
    6. **💬 Strictly Grounded Q&A**: Answers questions using only the uploaded PDF, formatting comparisons into tables and explicitly stating when information is absent.

    ### 🧠 AI & NLP Architecture
    - **Google Gemini 1.5 Flash**: Cloud Generative AI for structured simplification and grounded comprehension.
    - **Local Demonstration Engine**: High-accuracy extractive sentence scoring, term frequency analysis, and rule-based Cloze generation when offline.
    """)

    st.markdown("---")
    st.caption("College Mini-Capstone Project • Tailored for Average College Students")
