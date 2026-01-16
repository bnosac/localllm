"""
localllm: Run a local LLM alongside dspy
"""

__version__ = "0.1.0"


from .dspy import LocalLLM
from .models import localllm_download_model

__all__ = ["LocalLLM", "localllm_download_model"]
