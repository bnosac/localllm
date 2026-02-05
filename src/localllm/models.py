import os
import urllib.request
from pathlib import Path
from typing import Optional


def localllm_download_model(type: str = "gemma-3-270m-it-Q8_0", model_dir: Optional[str] = None, overwrite: bool = False, trace: bool = True) -> str:
    """
    Download a local gguf LLM model

    Parameters
    ----------
    type : str, default="gemma-3-270m-it-Q8_0"
        The type/name of the model to download. Currently supports:

        - "gemma-3-270m-it-Q8_0": Google Gemma 3 270M it model (Q8_0 quantization)
        - "gemma-3-270m-it-qat-Q4_0": Google Gemma 3 270M it model (Q4_0 quantization)
        - "gemma-3-1b-it-Q8_0": Google Gemma 3 1B it model (Q8_0 quantization)
        - "gemma-3-4b-it-Q4_K_M": Google Gemma 3 4b it model (Q4_K_M quantization)     
        - "gemma-3-12b-it-qat-Q4_0": Google Gemma 3 12B it model (Q4_0 quantization)
        - "GLM-4.6V-Flash-Q4_K_M": GLM 4.6V Flash model (Q4_K_M quantization)
        - "translategemma-4b-it-q8_0": TranslateGemma 4B it model (Q8_0 quantization)
        - "LFM2.5-1.2B-Instruct-Q4_K_M": LFM2.5 1.2B Instruct model (Q4_K_M quantization)
        - "Qwen3-8B-Q4_K_M": Qwen 3 8B model (Q4_K_M quantization)
        

    model_dir : str or None, default=None
        Directory where the model should be stored. If None, uses the path set in environment variable LOCALLLM_MODEL_DIR
        and if that environment variable does not exist, uses the default directory in the user's home folder (~/.localllm/models/).

    overwrite : bool, default=False
        If True, re-download the model even if it already exists.
        If False, skip download if the model file is already present.

    trace : bool, default=True
        If True, shows download progress.

    Returns
    -------
    str
        Path to the downloaded model file.

    Examples
    --------

    >>> from localllm import localllm_download_model
    >>> model_path = localllm_download_model("gemma-3-270m-it-Q8_0")                      # doctest: +SKIP
    >>> model_path = localllm_download_model("gemma-3-270m-it-Q8_0", overwrite=True)
    Downloading...
    >>> model_path = localllm_download_model("gemma-3-270m-it-Q8_0", overwrite=False)
    >>> model_path = localllm_download_model("not-existing-model")                        # doctest: +SKIP
    >>>
    >>> import os
    >>> path = os.getcwd()
    >>> model_path = localllm_download_model(model_dir=path)
    Downloading...
    >>> ## More models
    >>> os.remove(model_path)  # Clean up downloaded file
    >>> model_path = localllm_download_model("gemma-3-270m-it-qat-Q4_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-3-1b-it-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-3-4b-it-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-3-12b-it-qat-Q4_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("GLM-4.6V-Flash-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("translategemma-4b-it-q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("LFM2.5-1.2B-Instruct-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3-8B-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)

    Notes
    -----
    The function creates the target directory if it doesn't exist.
    Progress is displayed during download.
    """

    # Model registry - maps model types to their download URLs
    model_registry = {
        "gemma-3-270m-it-Q8_0": {
            "url": "https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q8_0.gguf",
            "filename": "google_gemma-3-270m-it-Q8_0.gguf",
        },
        "gemma-3-270m-it-qat-Q4_0": {
            "url": "https://huggingface.co/ggml-org/gemma-3-270m-it-qat-GGUF/resolve/main/gemma-3-270m-it-qat-Q4_0.gguf",
            "filename": "gemma-3-270m-it-qat-Q4_0.gguf",
        },
        "gemma-3-1b-it-Q8_0": {
            "url": "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q8_0.gguf",
            "filename": "google_gemma-3-1b-it-Q8_0.gguf",
        },
        "gemma-3-4b-it-Q4_K_M": {
            "url": "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
            #"url": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
            "filename": "google_gemma-3-4b-it-Q4_K_M.gguf",
        },
        "gemma-3-12b-it-qat-Q4_0": {
            "url": "https://huggingface.co/bartowski/google_gemma-3-12b-it-qat-GGUF/resolve/main/google_gemma-3-12b-it-qat-Q4_0.gguf",            
            "filename": "google_gemma-3-12b-it-qat-Q4_0.gguf",
        },           
        "GLM-4.6V-Flash-Q4_K_M": {
            "url": "https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF/resolve/main/GLM-4.6V-Flash-Q4_K_M.gguf",
            "filename": "GLM-4.6V-Flash-Q4_K_M.gguf",
        },
        "translategemma-4b-it-q8_0": {
            "url": "https://huggingface.co/NikolayKozloff/translategemma-4b-it-Q8_0-GGUF/resolve/main/translategemma-4b-it-q8_0.gguf",
            "filename": "translategemma-4b-it-q8_0.gguf",
        },
        "LFM2.5-1.2B-Instruct-Q4_K_M": {
            "url": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            "filename": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        },      
        "Qwen3-8B-Q4_K_M": {
            "url": "https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf",            
            "filename": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        },                      
    }

    # Validate model type
    if type not in model_registry:
        available = ", ".join(model_registry.keys())
        raise ValueError(f"Unknown model type '{type}'. Available models: {available}")

    model_info = model_registry[type]

    # Determine download directory
    if model_dir is None:
        # Check for environment variable first
        env_dir = os.environ.get("LOCALLLM_MODEL_DIR")
        if env_dir:
            model_dir = env_dir
        else:
            model_dir = os.path.join(Path.home(), ".localllm", "models")

    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Full path to model file
    model_path = os.path.join(model_dir, model_info["filename"])

    # Check if file already exists
    if os.path.exists(model_path) and not overwrite:
        return model_path
    if os.path.exists(model_path) and overwrite:
        os.remove(model_path)

    def download_progress(block_num, block_size, total_size):
        """Display download progress."""
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    try:
        # Download the model
        if trace:            
            print(f"Downloading {type} model...")
            print(f"From: {model_info['url']}")
            print(f"To: {model_path}")
            urllib.request.urlretrieve(model_info["url"], model_path, reporthook=download_progress)
        else:
            urllib.request.urlretrieve(model_info["url"], model_path)
        return model_path

    except Exception as e:
        print(f"\nError downloading model: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            os.remove(model_path)
        raise
