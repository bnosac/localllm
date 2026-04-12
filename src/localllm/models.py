import os
import urllib.request
from pathlib import Path
from typing import Optional

def localllm_list_models():
    """
    List all available models which you can easily download

    Returns
    -------
    dict
        a dictionary with all models, where each dictionary element contains source, url, filename

    Examples
    --------

    >>> from localllm import localllm_list_models
    >>> x = localllm_list_models()
    >>> list(x.keys())
    ['gemma-3-270m-it-qat-Q4_0', 'gemma-3-270m-it-Q8_0', 'gemma-3-1b-it-Q8_0', 'gemma-3-4b-it-qat-Q4_0', 'gemma-3-4b-it-Q4_K_M', 'gemma-3-12b-it-qat-Q4_0', 'GLM-4.6V-Flash-Q4_K_M', 'translategemma-4b-it-q8_0', 'translategemma-12b-it-q4_k_m', 'LFM2.5-350M-Q8_0', 'LFM2.5-1.2B-Instruct-Q4_K_M', 'LFM2.5-1.2B-Instruct-Q8_0', 'Qwen3-4B-Instruct-Q4_K_M', 'Qwen3-8B-Q4_K_M', 'Qwen3.5-0.8B-Q8_0', 'Qwen3.5-2B-Q4_K_M', 'Qwen3.5-4B-Q4_K_M', 'Qwen3.5-9B-Q4_K_M', 'gemma-4-e2b-it-Q8_0', 'gemma-4-E2B-it-Q4_K_M', 'gemma-4-e4b-it-Q8_0', 'gemma-4-E4B-it-Q4_K_M']
    >>> x['gemma-4-e2b-it-Q8_0']
    {'source': 'ggml-org', 'url': 'https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf', 'filename': 'gemma-4-e2b-it-Q8_0.gguf'}
    >>>
    >>> import pandas as pd                               # doctest: +SKIP
    >>> x = pd.DataFrame.from_dict(x, orient = "index")   # doctest: +SKIP
    """
    model_registry = {
        "gemma-3-270m-it-qat-Q4_0": {
            "source": "ggml-org",
            "url": "https://huggingface.co/ggml-org/gemma-3-270m-it-qat-GGUF/resolve/main/gemma-3-270m-it-qat-Q4_0.gguf",
            "filename": "gemma-3-270m-it-qat-Q4_0.gguf",
        },
        "gemma-3-270m-it-Q8_0": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q8_0.gguf",
            "filename": "google_gemma-3-270m-it-Q8_0.gguf",
        },        
        "gemma-3-1b-it-Q8_0": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q8_0.gguf",
            "filename": "google_gemma-3-1b-it-Q8_0.gguf",
        },
        "gemma-3-4b-it-qat-Q4_0": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/google_gemma-3-4b-it-qat-GGUF/resolve/main/google_gemma-3-4b-it-qat-Q4_0.gguf",
            "filename": "google_gemma-3-4b-it-qat-Q4_0.gguf",
        },
        "gemma-3-4b-it-Q4_K_M": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
            # "url": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
            "filename": "google_gemma-3-4b-it-Q4_K_M.gguf",
        },
        "gemma-3-12b-it-qat-Q4_0": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/google_gemma-3-12b-it-qat-GGUF/resolve/main/google_gemma-3-12b-it-qat-Q4_0.gguf",
            "filename": "google_gemma-3-12b-it-qat-Q4_0.gguf",
        },
        "GLM-4.6V-Flash-Q4_K_M": {
            "source": "ggml-org",
            "url": "https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF/resolve/main/GLM-4.6V-Flash-Q4_K_M.gguf",
            "filename": "GLM-4.6V-Flash-Q4_K_M.gguf",
        },
        "translategemma-4b-it-q8_0": {
            "source": "NikolayKozloff",
            "url": "https://huggingface.co/NikolayKozloff/translategemma-4b-it-Q8_0-GGUF/resolve/main/translategemma-4b-it-q8_0.gguf",
            "filename": "translategemma-4b-it-q8_0.gguf",
        },
        "translategemma-12b-it-q4_k_m": {
            "source": "NikolayKozloff",
            "url": "https://huggingface.co/NikolayKozloff/translategemma-12b-it-Q4_K_M-GGUF/resolve/main/translategemma-12b-it-q4_k_m.gguf",
            "filename": "translategemma-12b-it-q4_k_m.gguf",
        },        
        "LFM2.5-350M-Q8_0": {
            "source": "LiquidAI",
            "url": "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf",
            "filename": "LFM2.5-350M-Q8_0.gguf",
        },        
        "LFM2.5-1.2B-Instruct-Q4_K_M": {
            "source": "LiquidAI",
            "url": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            "filename": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        },
        "LFM2.5-1.2B-Instruct-Q8_0": {
            "source": "LiquidAI",
            "url": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf",
            "filename": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        },
        "Qwen3-4B-Instruct-Q4_K_M": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            "filename": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        },        
        "Qwen3-8B-Q4_K_M": {
            "source": "bartowski",
            "url": "https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf",
            "filename": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        },
        "Qwen3.5-0.8B-Q8_0": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
            "filename": "Qwen3.5-0.8B-Q8_0.gguf",
        },    
        "Qwen3.5-2B-Q4_K_M": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
            "filename": "Qwen3.5-2B-Q4_K_M.gguf",
        }, 
        "Qwen3.5-4B-Q4_K_M": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
            "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        },                       
        "Qwen3.5-9B-Q4_K_M": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
            "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        }, 
        "gemma-4-e2b-it-Q8_0": {
            "source": "ggml-org",
            "url": "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf",
            "filename": "gemma-4-e2b-it-Q8_0.gguf",
        },
        "gemma-4-E2B-it-Q4_K_M": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
            "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        },  
        "gemma-4-e4b-it-Q8_0": {
            "source": "ggml-org",
            "url": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-e4b-it-Q8_0.gguf",
            "filename": "gemma-4-e4b-it-Q8_0.gguf",
        },                  
        "gemma-4-E4B-it-Q4_K_M": {
            "source": "unsloth",
            "url": "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
            "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        },                            
    }
    return model_registry


def localllm_download_model(type: str = "gemma-3-270m-it-Q8_0", model_dir: Optional[str] = None, overwrite: bool = False, trace: bool = True) -> str:
    """
    Download a local gguf LLM model

    Parameters
    ----------
    type : str, default="gemma-3-270m-it-Q8_0"
        The type/name of the model to download. Currently supports the following models. You can either give the model name or localllm/{model_name}

        - "gemma-3-270m-it-qat-Q4_0": Google Gemma 3 270M it model (Q4_0 quantization)
        - "gemma-3-270m-it-Q8_0": Google Gemma 3 270M it model (Q8_0 quantization)
        - "gemma-3-1b-it-Q8_0": Google Gemma 3 1B it model (Q8_0 quantization)
        - "gemma-3-4b-it-qat-Q4_0": Google Gemma 3 4b it model (Q4_0 quantization - with quantized aware training)
        - "gemma-3-4b-it-Q4_K_M": Google Gemma 3 4b it model (Q4_K_M quantization)
        - "gemma-3-12b-it-qat-Q4_0": Google Gemma 3 12B it model (Q4_0 quantization)
        - "GLM-4.6V-Flash-Q4_K_M": GLM 4.6V Flash model (Q4_K_M quantization)
        - "translategemma-4b-it-q8_0": TranslateGemma 4B it model (Q8_0 quantization)
        - "translategemma-12b-it-q4_k_m": TranslateGemma 12B it model (Q4_K_M quantization)
        - "LFM2.5-350M-Q8_0": LFM2.5 350M model (Q8_0 quantization)  
        - "LFM2.5-1.2B-Instruct-Q4_K_M": LFM2.5 1.2B Instruct model (Q4_K_M quantization)
        - "LFM2.5-1.2B-Instruct-Q8_0": LFM2.5 1.2B Instruct model (Q8_0 quantization)
        - "Qwen3-4B-Instruct-Q4_K_M": Qwen 3 4B model (Q4_K_M quantization)
        - "Qwen3-8B-Q4_K_M": Qwen 3 8B model (Q4_K_M quantization)
        - "Qwen3.5-0.8B-Q8_0": Qwen 3.5 0.8B model (Q8_0 quantization)
        - "Qwen3.5-2B-Q4_K_M": Qwen 3.5 2B model (Q4_K_M quantization)
        - "Qwen3.5-4B-Q4_K_M": Qwen 3.5 4B model (Q4_K_M quantization)
        - "Qwen3.5-9B-Q4_K_M": Qwen 3.5 9B model (Q4_K_M quantization)
        - "gemma-4-e2b-it-Q8_0":   Google Gemma 4 E2B Instruct model (Q8_0 quantization)
        - "gemma-4-E2B-it-Q4_K_M": Google Gemma 4 E2B Instruct model (Q4_K_M quantization)
        - "gemma-4-e4b-it-Q8_0":   Google Gemma 4 E4B Instruct model (Q8_0 quantization)
        - "gemma-4-E4B-it-Q4_K_M": Google Gemma 4 E4B Instruct model (Q4_K_M quantization)

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
    >>> model_path = localllm_download_model("gemma-3-4b-it-qat-Q4_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-3-4b-it-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-3-12b-it-qat-Q4_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("GLM-4.6V-Flash-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("translategemma-4b-it-q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("translategemma-12b-it-q4_k_m", overwrite=True, trace = False)
    >>> os.remove(model_path)    
    >>> model_path = localllm_download_model("LFM2.5-350M-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)        
    >>> model_path = localllm_download_model("LFM2.5-1.2B-Instruct-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("LFM2.5-1.2B-Instruct-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3-4B-Instruct-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)    
    >>> model_path = localllm_download_model("Qwen3-8B-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3.5-0.8B-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3.5-2B-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3.5-4B-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("Qwen3.5-9B-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)
    >>> model_path = localllm_download_model("gemma-4-e2b-it-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path) 
    >>> model_path = localllm_download_model("gemma-4-E2B-it-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)     
    >>> model_path = localllm_download_model("gemma-4-e4b-it-Q8_0", overwrite=True, trace = False)
    >>> os.remove(model_path)     
    >>> model_path = localllm_download_model("gemma-4-E4B-it-Q4_K_M", overwrite=True, trace = False)
    >>> os.remove(model_path)            
    
    Notes
    -----
    The function creates the target directory if it doesn't exist.
    Progress is displayed during download.
    """
    if type.startswith("localllm/"):
        type = type.replace("localllm/", "")

    # Model registry - maps model types to their download URLs
    model_registry = localllm_list_models()

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
