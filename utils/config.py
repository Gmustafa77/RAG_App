import os
from dataclasses import dataclass
from dotenv import load_dotenv

# load_dotenv() reads the ".env" file in the project root and copies its
# key=value pairs into the process's environment variables (os.environ).
# This keeps secrets (like API keys) OUT of the source code.
load_dotenv()


@dataclass
class Config:
    """
    A simple, typed container for all app settings.

    Using a dataclass (instead of scattering os.getenv() calls everywhere)
    gives us:
      - autocomplete / type hints in your editor
      - one obvious place to see every setting the app depends on
      - easy defaults if a .env value is missing
    """

    # --- Provider switch ------------------------------------------------------
    # Which provider generates CHAT completions. One of: "openai", "azure", "gemini"
    # (kept independent from embedding_provider so you can mix and match).
    chat_provider: str = os.getenv("CHAT_PROVIDER", "").lower()
    # Backward-compat: USE_AZURE_OPENAI=true still works if CHAT_PROVIDER isn't set.
    use_azure_openai: bool = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    # Which provider generates EMBEDDINGS. This is independent of chat, so you
    # can e.g. keep chat on Azure/OpenAI but use free Gemini embeddings.
    # Valid values: "openai", "azure", "gemini"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    # --- OpenAI credentials & model choices (plain OpenAI) -------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "api_key")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # --- Azure OpenAI credentials & deployment names --------------------------
    # Azure doesn't use bare model names like "gpt-4o" at request time — it
    # routes by the DEPLOYMENT NAME you chose when you deployed the model in
    # Azure AI Studio. The key, endpoint, and api version all come from the
    # Azure Portal resource page ("Keys and Endpoint").
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    # --- Google Gemini credentials & embedding model (free tier) -------------
    # Get a free key (no card required) at https://aistudio.google.com/apikey
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

    # --- Vector store (ChromaDB) settings ------------------------------------
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "aetna_insurance_docs")

    # --- Chunking settings ----------------------------------------------------
    # CHUNK_SIZE: how many characters go into each chunk of text.
    # CHUNK_OVERLAP: how many characters two consecutive chunks share, so a
    #                sentence that spans a chunk boundary isn't lost entirely.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # --- Retrieval settings -----------------------------------------------------
    # RETRIEVAL_TOP_K: how many of the most-similar chunks we hand to the LLM
    # as context for each question.
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 5))

    # --- File storage paths -------------------------------------------------
    pdf_upload_dir: str = os.getenv("PDF_UPLOAD_DIR", "data/pdfs")

    # def validate(self) -> None:
    #     """
    #     Raise a clear, human-readable error early if required settings are
    #     missing, instead of letting the app crash later with a cryptic
    #     "401 Unauthorized" deep inside the OpenAI SDK.
    #     """
    @property
    def resolved_chat_provider(self) -> str:
        """
        The effective chat provider, honoring CHAT_PROVIDER first and
        falling back to the older USE_AZURE_OPENAI boolean for backward
        compatibility with existing .env files.
        """
        if self.chat_provider in ("openai", "azure", "gemini"):
            return self.chat_provider
        return "azure" if self.use_azure_openai else "openai"

    def validate(self) -> None:
        chat_provider = self.resolved_chat_provider

        # --- Chat provider check ---
        if chat_provider == "azure":
            missing = [
                name
                for name, value in [
                    ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
                    ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                    ("AZURE_OPENAI_CHAT_DEPLOYMENT", self.azure_openai_chat_deployment),
                ]
                if not value
            ]
            if missing:
                raise ValueError(
                    "Chat provider is 'azure' but these .env values are missing: "
                    + ", ".join(missing)
                )
        elif chat_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError(
                    "CHAT_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                    "Get a free key at https://aistudio.google.com/apikey"
                )
        elif not self.openai_api_key or self.openai_api_key == "api_key":
            raise ValueError(
                "Chat provider is 'openai' but OPENAI_API_KEY is not set (or "
                "is still the placeholder 'api_key'). Add a real key from "
                "https://platform.openai.com/api-keys, or set "
                "CHAT_PROVIDER=azure or CHAT_PROVIDER=gemini in .env."
            )

        # --- Embedding provider check ---
        if self.embedding_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError(
                    "EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                    "Get a free key at https://aistudio.google.com/apikey"
                )
        elif self.embedding_provider == "azure":
            missing = [
                name
                for name, value in [
                    ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
                    ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                    ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", self.azure_openai_embedding_deployment),
                ]
                if not value
            ]
            if missing:
                raise ValueError(
                    "EMBEDDING_PROVIDER=azure but these .env values are missing: "
                    + ", ".join(missing)
                )
        elif self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
                )
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER='{self.embedding_provider}'. "
                "Must be one of: openai, azure, gemini."
            )


# A single, shared Config instance that every other module can import:
#     from utils.config import settings
settings = Config()
