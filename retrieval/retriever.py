# ==============================================================================
# retrieval/retriever.py
# ------------------------------------------------------------------------------
# STEP 6 & 7 of the RAG pipeline: "Perform similarity search" +
# "Retrieve top matching chunks"
#
# WHAT HAPPENS HERE (GenAI beginners):
#   1. The user types a question, e.g. "Do I need a referral to see a
#      specialist?"
#   2. We turn THAT question into an embedding vector using the exact same
#      embedding model we used on the documents (embeddings/embedding_service.py).
#   3. ChromaDB compares that question-vector against every stored
#      chunk-vector and returns the ones that are mathematically closest —
#      i.e., most similar in MEANING, not just shared keywords.
#   4. We return the top-K matches (K = settings.retrieval_top_k) along with
#      their source metadata (filename + page number), which downstream the
#      rag_pipeline.py uses to build the LLM prompt and citations.
#
# WHY THIS IS A SEPARATE MODULE FROM vectorstore/chroma_manager.py:
#   chroma_manager.py is about STORAGE (how vectors get in and out of the
#   database). retriever.py is about the QUESTION-ANSWERING-TIME behavior:
#   taking a live user question and turning it into ranked, usable context.
#   Keeping these separate makes each module easier to test and reason about.
# ==============================================================================

from dataclasses import dataclass
from typing import List

from vectorstore.chroma_manager import ChromaManager
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    One retrieved chunk of context, ready to be handed to the LLM and later
    cited in the Streamlit UI.
    """

    text: str
    source_file: str
    page_number: int
    similarity_score: float  # lower = more similar (Chroma uses distance, not %)


class Retriever:
    """
    Wraps ChromaManager to provide a simple, purpose-built API for
    "given a user's question, give me the most relevant document chunks."
    """

    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager

    def retrieve(self, question: str, top_k: int = None) -> List[RetrievedChunk]:
        """
        Run similarity search for a user's question and return the top-K
        most relevant chunks, each tagged with its source for citation.

        Args:
            question: the user's natural-language question
            top_k: how many chunks to retrieve (defaults to config value)

        Returns:
            A list of RetrievedChunk objects, ranked from most to least
            similar.
        """
        k = top_k or settings.retrieval_top_k
        logger.info(f"Retrieving top {k} chunks for question: '{question}'")

        # similarity_search_with_scores() does two things under the hood:
        #   (a) embeds `question` using our embedding model
        #   (b) asks ChromaDB for the k nearest chunk-vectors to it
        results = self.chroma_manager.similarity_search_with_scores(question, top_k=k)

        retrieved_chunks: List[RetrievedChunk] = []
        for document, score in results:
            retrieved_chunks.append(
                RetrievedChunk(
                    text=document.page_content,
                    source_file=document.metadata.get("source", "unknown"),
                    page_number=document.metadata.get("page", 0),
                    similarity_score=float(score),
                )
            )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks

    def has_documents(self) -> bool:
        """
        Check whether ANY documents have been uploaded and indexed yet.
        Used by the Streamlit UI to show a friendly "please upload a
        document first" message instead of silently returning empty answers.
        """
        return self.chroma_manager.document_count() > 0
