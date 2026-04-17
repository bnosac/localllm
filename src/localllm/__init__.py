"""
localllm: Run a local LLM alongside dspy
"""

__version__ = "0.3.0"


from .dspy import LocalLLM, LocalChatAdapter, dspy_signature_defaults
from .models import localllm_download_model, localllm_list_models
from .utils import txt_locate_all, txt_locate, merge_spans
from .config import localllm_connect
from .config import localllm_connect as connect

__all__ = [
    "localllm_connect",
    "LocalLLM", "localllm_download_model", "localllm_list_models", "LocalChatAdapter", "dspy_signature_defaults",
    "txt_locate_all", "txt_locate", "merge_spans"]
