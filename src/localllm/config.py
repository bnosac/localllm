
import os
import dspy
from .dspy import LocalLLM, LocalChatAdapter
from .models import localllm_download_model

def localllm_use(
        lm = "localllm/gemma-3-270m-it-Q8_0", 
        model_kwargs = {"n_ctx": 32768, "n_gpu_layers": -1, "n_threads": 1, "flash_attn": True, "verbose": False}, 
        adapter = LocalChatAdapter()) -> None:
    if isinstance(lm, str):        
        if lm.startswith("localllm/"):
            model_name = lm.replace("localllm/", "")
            model_name = localllm_download_model(model_name)
            ## Case where you specify localllm/gemma-3-4b-it-Q4_K_M for example
        else:
            model_name = lm        
        if os.path.exists(model_name):
            ## Local gguf LLM model
            from llama_cpp import Llama
            transformer = Llama(model_path=model_name, **model_kwargs)
            out = dspy.configure(lm = LocalLLM(transformer), adapter = adapter)
        else:
            ## Other API service providers
            lm = dspy.LM(model_name, **model_kwargs)
            out = dspy.configure(lm = lm, adapter = adapter)            
        return out
    else:
        raise NotImplementedError("lm can only be of type str for now")
    
