# ==============================================================================
# rag/rag_pipeline.py
# ------------------------------------------------------------------------------
# This is the ORCHESTRATOR of the whole RAG pipeline. It ties together
# every step, Steps 6 through 9:
#     Step 6: Perform similarity search        (via retrieval/retriever.py)
#     Step 7: Retrieve top matching chunks      (via retrieval/retriever.py)
#     Step 8: Send retrieved chunks to the LLM  (via prompts/prompt_template.py)
#     Step 9: Generate a grounded response      (via OpenAI GPT-4o)
#
# WHAT IS "RAG" IN ONE SENTENCE? (GenAI beginners)
#   Retrieval-Augmented Generation = instead of asking an LLM to answer
#   purely from what it memorized during training, we first RETRIEVE the
#   most relevant snippets of TRUSTED, up-to-date documents, and then ask
#   the LLM to GENERATE an answer using ONLY those snippets. This is what
#   lets a general-purpose model like GPT-4o answer accurately about a
#   SPECIFIC Aetna plan's specific rules — something it was never trained on.
#
# HOW LANGCHAIN IS USED HERE:
#   We use LangChain's ChatOpenAI wrapper to call GPT-4o. LangChain
#   standardizes how you send "messages" (system/user/assistant turns) to
#   different LLM providers, so if you ever wanted to swap GPT-4o for
#   another model, most of this code wouldn't need to change.
#
# ALSO PROVIDES: ingest_documents(), which runs Steps 1-5 (upload -> extract
# -> chunk -> embed -> store) for newly uploaded PDFs, before any questions
# can be answered.
# ==============================================================================

from dataclasses import dataclass, field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from chunking.text_splitter import DocumentChunker
from embeddings.embedding_service import EmbeddingService
from ingestion.pdf_loader import PDFLoader
from prompts.prompt_template import SYSTEM_PROMPT, build_user_prompt
from retrieval.retriever import Retriever, RetrievedChunk
from vectorstore.chroma_manager import ChromaManager
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_text(content) -> str:
    """
    Normalize an AIMessage's .content into a plain string.

    LangChain's .content is a plain string for OpenAI/Azure models, but
    some Gemini models return it as a LIST of structured content blocks
    instead — e.g. [{"type": "text", "text": "...", "extras": {...}}] —
    where "extras" can include internal metadata like reasoning signatures.
    Without this normalization, that whole raw block list gets displayed
    to the user instead of just the answer text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()

    return str(content)


@dataclass
class RAGResponse:
    """
    The final answer package returned to the Streamlit frontend: the
    generated text plus the exact chunks it was grounded in, so the UI can
    render citations underneath the chat bubble.
    """

    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    """
    The single entry point the Streamlit frontend talks to. It exposes just
    two methods:
      - ingest_documents(file_paths): run Steps 1-5 on newly uploaded PDFs
      - ask(question): run Steps 6-9 to answer a user's question
    """

    def __init__(self):
        settings.validate()

        # --- Shared building blocks, wired together once at startup -------
        self.pdf_loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.chroma_manager = ChromaManager(embedding_service=self.embedding_service)
        self.retriever = Retriever(self.chroma_manager)

        # temperature=0 makes the model as deterministic/factual as possible —
        # we want consistent, literal answers about deductibles, not creative
        # writing.
        chat_provider = settings.resolved_chat_provider

        if chat_provider == "gemini":
            # ChatGoogleGenerativeAI calls Google's free-tier Gemini API.
            # Reuses the same GEMINI_API_KEY as embeddings, if you're using
            # Gemini for both.
            self.llm = ChatGoogleGenerativeAI(
                model=settings.gemini_chat_model,
                google_api_key=settings.gemini_api_key,
                temperature=0,
            )
            self._chat_model_label = settings.gemini_chat_model
        elif chat_provider == "azure":
            # AzureChatOpenAI routes calls to your Azure resource's endpoint.
            # Azure identifies the model by DEPLOYMENT NAME, not "gpt-4o".
            self.llm = AzureChatOpenAI(
                azure_deployment=settings.azure_openai_chat_deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                temperature=0,
            )
            self._chat_model_label = settings.azure_openai_chat_deployment
        else:
            # ChatOpenAI is LangChain's wrapper around OpenAI's chat
            # completion API.
            self.llm = ChatOpenAI(
                model=settings.chat_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
            self._chat_model_label = settings.chat_model

        logger.info("RAGPipeline initialized and ready")

    # -------------------------------------------------------------------
    # INGESTION: Steps 1 -> 5 (upload, extract, chunk, embed, store)
    # -------------------------------------------------------------------
    def ingest_documents(self, file_paths: List[str]) -> int:
        """
        Run the full ingestion pipeline on a list of uploaded PDF file paths.

        Step 1 (Upload) happens in the Streamlit UI (frontend/streamlit_ui.py),
        which saves uploaded files to disk and passes their paths here.

        Args:
            file_paths: paths to PDF files already saved on disk, e.g.
                        ["data/pdfs/Aetna_EOC_2026.pdf"]

        Returns:
            The number of chunks successfully embedded and stored.
        """
        logger.info(f"Starting ingestion for {len(file_paths)} file(s)")

        # Step 2: Extract text from PDFs
        pages = self.pdf_loader.load_multiple_pdfs(file_paths)
        if not pages:
            logger.warning("No text could be extracted from the uploaded PDF(s)")
            return 0

        # Step 3: Chunk documents
        chunks = self.chunker.chunk_pages(pages)

        # Step 4 + 5: Generate embeddings & store them in ChromaDB
        # (embedding happens INSIDE add_chunks(), via the embedding_function
        # that was wired into ChromaManager's Chroma vectorstore)
        stored_count = self.chroma_manager.add_chunks(chunks)

        logger.info(f"Ingestion complete: {stored_count} chunks stored")
        return stored_count

    # -------------------------------------------------------------------
    # QUESTION ANSWERING: Steps 6 -> 9 (retrieve, augment, generate)
    # -------------------------------------------------------------------
    def ask(self, question: str) -> RAGResponse:
        """
        Answer a user's question using only the uploaded documents.

        Args:
            question: the user's natural-language question

        Returns:
            A RAGResponse containing the generated answer text and the
            source chunks it was grounded in (for citation display).
        """
        # Guard clause: if nothing has been uploaded/indexed yet, don't
        # even call the LLM — just tell the user clearly what to do.
        if not self.retriever.has_documents():
            return RAGResponse(
                answer=(
                    "I don't have any insurance documents to search yet. "
                    "Please upload a PDF (like your Evidence of Coverage or "
                    "Summary of Benefits and Coverage) using the sidebar first."
                ),
                sources=[],
            )

        # Step 6 + 7: similarity search -> top-K retrieved chunks
        retrieved_chunks = self.retriever.retrieve(question)

        # Step 8: Send retrieved chunks + question to the LLM.
        # We build a "system" message (the ground rules, sent once, always
        # obeyed first) and a "user" message (context + question combined).
        user_prompt = build_user_prompt(question, retrieved_chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"Sending question to {self._chat_model_label} with {len(retrieved_chunks)} context chunks")

        # Step 9: Generate a grounded response.
        # self.llm.invoke() sends the messages to the LLM and returns an
        # AIMessage; _extract_text() pulls out just the answer text,
        # regardless of whether the provider returns .content as a plain
        # string (OpenAI/Azure) or a list of content blocks (some Gemini
        # models).
        ai_message = self.llm.invoke(messages)
        answer_text = _extract_text(ai_message.content)

        logger.info("Received grounded response from LLM")

        return RAGResponse(answer=answer_text, sources=retrieved_chunks)