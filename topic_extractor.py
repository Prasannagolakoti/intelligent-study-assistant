"""
Topic Extractor Module
Extracts key study concepts, frequency-weighted topics, and revision priorities.
Supports both LLM-powered semantic analysis and local statistical n-gram extraction.
"""

from typing import Dict, List, Optional, Any
from modules.study_assistant import StudyAssistant
from modules.local_demo import LocalStudyAssistant


class TopicExtractor:
    """
    Coordinates extraction of key concepts, technical topics, and suggested
    study priorities from student notes.
    """

    def __init__(self):
        self.assistant = StudyAssistant()
        self.local_engine = LocalStudyAssistant()

    def extract_topics(
        self,
        text: str,
        custom_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract ranked topics from text.
        Returns a list of structured topic dicts:
        - rank (int)
        - topic (str)
        - priority ('High Priority', 'Medium Priority', 'Foundational Concept')
        - priority_class ('high', 'medium', 'normal')
        - reason (str)
        - context_snippet (str)
        """
        if not text or not text.strip():
            return []

        # Delegates to StudyAssistant which attempts LLM first with local fallback
        return self.assistant.extract_important_topics(text, custom_key=custom_key)
