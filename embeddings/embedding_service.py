from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Thin wrapper around a LangChain embeddings client (OpenAI, Azure OpenAI,
    or Google Gemini — chosen via EMBEDDING_PROVIDER in .env).

    We centralize embedding creation here (rather than calling a provider
    SDK directly all over the codebase) so:
      - the provider + model name is configured in ONE place (utils/config.py)
      - both the ingestion pipeline (embedding document chunks) and the
        query pipeline (embedding a user's question) use the EXACT SAME
        model — this is critical, because comparing vectors from two
        different embedding models produces meaningless similarity scores.
    """

    def __init__(self):
        settings.validate()  # fail fast with a clear error if API key/config is missing

        provider = settings.embedding_provider

        if provider == "gemini":
            # GoogleGenerativeAIEmbeddings calls Google's free-tier Gemini
            # Embedding API. No card required to get a key — see
            # https://aistudio.google.com/apikey
            self.client = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.gemini_api_key,
            )
            logger.info(
                f"EmbeddingService initialized with Gemini model="
                f"'{settings.gemini_embedding_model}'"
            )
        elif provider == "azure":
            # AzureOpenAIEmbeddings routes calls to your Azure resource's
            # endpoint instead of api.openai.com. Azure identifies the model
            # by DEPLOYMENT NAME, not the bare model string.
            self.client = AzureOpenAIEmbeddings(
                azure_deployment=settings.azure_openai_embedding_deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            logger.info(
                f"EmbeddingService initialized with Azure deployment="
                f"'{settings.azure_openai_embedding_deployment}'"
            )
        else:
            # LangChain's OpenAIEmbeddings handles the actual HTTP calls to
            # OpenAI's /v1/embeddings endpoint, batching, and retries.
            self.client = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
            logger.info(f"EmbeddingService initialized with model='{settings.embedding_model}'")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a batch of text chunks into embedding vectors.
        Used during INGESTION (Step 4), once per uploaded document.

        Args:
            texts: list of chunk strings, e.g. ["The annual deductible is...", ...]

        Returns:
            A list of vectors (each vector is a list of floats), in the
            same order as the input texts.
        """
        logger.info(f"Embedding {len(texts)} document chunks...")
        vectors = self.client.embed_documents(texts)
        logger.info(f"Generated {len(vectors)} embeddings (dimension={len(vectors[0]) if vectors else 0})")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single piece of text — normally the USER'S QUESTION —
        into an embedding vector. Used during RETRIEVAL (Step 6), once per
        chat message the user sends.

        Args:
            text: the user's natural-language question, e.g.
                  "What's my copay for an ER visit?"

        Returns:
            A single embedding vector (list of floats).
        """
        return self.client.embed_query(text)

    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """
        Expose the underlying LangChain OpenAIEmbeddings object directly.
        Useful because LangChain's Chroma vector store wrapper (used in
        vectorstore/chroma_manager.py) wants an embeddings OBJECT, not just
        our embed_documents()/embed_query() convenience methods, so it can
        call embeddings internally whenever needed.
        """
        return self.client
