"""
localllm: Run a local LLM alongside dspy
"""

__version__ = "0.2.2"


from .dspy import LocalLLM, LocalChatAdapter, dspy_signature_defaults
from .models import localllm_download_model
from .utils import txt_locate_all, txt_locate, merge_spans
from .config import localllm_connect

__all__ = [
    "localllm_connect",
    "LocalLLM", "localllm_download_model", "LocalChatAdapter", "dspy_signature_defaults",
    "txt_locate_all", "txt_locate", "merge_spans"]
