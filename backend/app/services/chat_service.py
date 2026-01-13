"""
Chat Service - Semantic Q&A for Research Papers
Uses RAG (Retrieval Augmented Generation) to answer questions about papers
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

from app.data_models.schemas import ChatMessage, ChatResponse, ChatSource

# Add semantic-document-search to path
semantic_search_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../semantic-document-search/src")
)
if semantic_search_path not in sys.path:
    sys.path.insert(0, semantic_search_path)

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for handling Q&A chat about research papers.
    Uses semantic search + LLM for intelligent responses.
    """

    def __init__(self):
        self._pipeline = None
        self._qa_pipeline = None
        self._groq_llm = None
        self._initialized = False
        self._job_contexts: Dict[str, Dict] = {}  # Cache paper contexts per job

    def _lazy_init(self):
        """Lazy initialization of heavy components"""
        if self._initialized:
            return

        try:
            from integrated_pipeline import DocumentSearchPipeline
            from simple_qa_pipeline import SimpleQAPipeline

            logger.info("Initializing semantic search pipeline for chat...")

            self._pipeline = DocumentSearchPipeline(
                persist_directory="./data/chroma_chat_db",
                collection_name="paper_chat",
                use_grobid=False,  # We already have processed text
                chunk_size=800,
                chunk_overlap=100,
            )

            self._qa_pipeline = SimpleQAPipeline(self._pipeline, max_context_chunks=5)

            # Try to initialize Groq LLM if API key is available
            try:
                from groq_llm_interface import GroqLLMInterface

                self._groq_llm = GroqLLMInterface()
                logger.info("Groq LLM initialized for enhanced responses")
            except Exception as e:
                logger.warning(f"Groq LLM not available, using simple QA: {e}")
                self._groq_llm = None

            self._initialized = True
            logger.info("Chat service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise

    def index_paper_for_chat(self, job_id: str, paper_data: Dict[str, Any]) -> bool:
        """
        Index a paper's content for semantic search chat.
        Called after paper analysis is complete.

        Args:
            job_id: The analysis job ID
            paper_data: Parsed paper data with sections

        Returns:
            True if indexing succeeded
        """
        self._lazy_init()

        try:
            # Build text chunks from paper sections
            chunks = []

            # Add abstract
            if paper_data.get("abstract"):
                chunks.append(
                    {
                        "text": paper_data["abstract"],
                        "metadata": {
                            "job_id": job_id,
                            "section": "Abstract",
                            "source": paper_data.get("title", "Unknown"),
                        },
                    }
                )

            # Add sections
            for section_name, content in paper_data.get("sections", {}).items():
                if content and len(content) > 50:
                    # Split large sections into smaller chunks
                    section_chunks = self._chunk_text(content, max_size=800)
                    for i, chunk_text in enumerate(section_chunks):
                        chunks.append(
                            {
                                "text": chunk_text,
                                "metadata": {
                                    "job_id": job_id,
                                    "section": section_name,
                                    "chunk_index": i,
                                    "source": paper_data.get("title", "Unknown"),
                                },
                            }
                        )

            # Store paper context for quick access
            self._job_contexts[job_id] = {
                "title": paper_data.get("title", "Unknown"),
                "abstract": paper_data.get("abstract", ""),
                "sections": list(paper_data.get("sections", {}).keys()),
                "chunk_count": len(chunks),
            }

            # Add to vector store
            if chunks:
                self._pipeline.vector_store.add_documents_batch(
                    texts=[c["text"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks],
                    ids=[f"{job_id}_{i}" for i in range(len(chunks))],
                )

            logger.info(f"Indexed {len(chunks)} chunks for job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index paper for chat: {e}")
            return False

    def _chunk_text(self, text: str, max_size: int = 800) -> List[str]:
        """Split text into smaller chunks"""
        if len(text) <= max_size:
            return [text]

        chunks = []
        sentences = text.replace("\n", " ").split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def ask_question(
        self,
        job_id: str,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> ChatResponse:
        """
        Answer a question about a specific paper.

        Args:
            job_id: The paper's job ID
            question: User's question
            conversation_history: Previous messages for context

        Returns:
            ChatResponse with answer and sources
        """
        self._lazy_init()

        try:
            # Search for relevant chunks from this paper
            search_filter = {"job_id": job_id}

            results = self._pipeline.vector_store.search(
                query=question, n_results=5, where=search_filter
            )

            if not results or not results.get("documents"):
                return ChatResponse(
                    message="I don't have enough information about this paper to answer your question. The paper may not have been fully indexed.",
                    sources=[],
                    confidence=0.0,
                )

            # Build context from results
            context_parts = []
            sources = []

            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results.get("documents", [[]])[0],
                    results.get("metadatas", [[]])[0],
                    results.get("distances", [[]])[0],
                )
            ):
                similarity = 1 - (distance / 2)  # Convert distance to similarity
                if similarity > 0.3:  # Only use relevant chunks
                    context_parts.append(
                        f"[{metadata.get('section', 'Section')}]: {doc}"
                    )
                    sources.append(
                        ChatSource(
                            section=metadata.get("section", "Unknown"),
                            similarity=round(similarity, 2),
                        )
                    )

            context = "\n\n".join(context_parts)

            # Generate response
            if self._groq_llm and context:
                # Use Groq LLM for high-quality response
                response = self._groq_llm.generate_response(
                    question=question,
                    context=context,
                    system_prompt=self._get_system_prompt(job_id, conversation_history),
                )
                answer = response.get("response", response.get("text", ""))
                confidence = min(
                    0.95,
                    sum(s.similarity for s in sources) / len(sources) if sources else 0,
                )
            else:
                # Fallback to simple context-based answer
                answer = self._generate_simple_answer(question, context_parts)
                confidence = (
                    sum(s.similarity for s in sources) / len(sources) if sources else 0
                )

            return ChatResponse(
                message=answer,
                sources=sources[:3],  # Top 3 sources
                confidence=round(confidence, 2),
            )

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return ChatResponse(
                message="I encountered an error while processing your question. Please try again.",
                sources=[],
                confidence=0.0,
            )

    def _get_system_prompt(
        self, job_id: str, history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Generate system prompt with paper context"""
        paper_info = self._job_contexts.get(job_id, {})

        prompt = f"""You are an expert research assistant helping users understand a specific academic paper.

Paper Title: {paper_info.get("title", "Unknown")}
Available Sections: {", ".join(paper_info.get("sections", []))}

Guidelines:
- Answer based ONLY on the provided context from the paper
- Be precise and cite specific sections when possible
- If the context doesn't contain the answer, say so clearly
- Use academic but accessible language
- For technical terms, provide brief explanations
- Keep responses concise but comprehensive"""

        return prompt

    def _generate_simple_answer(self, question: str, context_parts: List[str]) -> str:
        """Generate a simple answer without LLM"""
        if not context_parts:
            return "I couldn't find relevant information to answer your question."

        # Find most relevant sentence
        question_words = set(question.lower().split())
        best_match = ""
        best_score = 0

        for part in context_parts:
            sentences = part.split(". ")
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words & sentence_words)
                if overlap > best_score:
                    best_score = overlap
                    best_match = sentence

        if best_match:
            return f"Based on the paper: {best_match.strip()}"

        # Return first context as fallback
        return f"Here's relevant information from the paper:\n\n{context_parts[0][:500]}..."


# Singleton instance
chat_service = ChatService()
