import os
import dspy
from .dspy import LocalLLM, LocalChatAdapter
from .models import localllm_download_model


def localllm_connect(
    lm: str = "localllm/gemma-3-270m-it-Q8_0",
    model_kwargs: dict = {"n_ctx": 32768, "n_gpu_layers": -1, "n_threads": 1, "flash_attn": True, "verbose": False},
    adapter=None,
    trace: bool = False,
):
    """
    Connect to a LocalLLM model

    Parameters
    ----------
    lm : str, default="localllm/gemma-3-270m-it-Q8_0"
        The name of the localllm model.
        Passed on to localllm_download_model if the string starts with "localllm". If not uses dspy.LM to connect to an API.

    model_kwargs : dict
        A dictionary of model arguments passed on to Llama in case of a localllm model or passed on to dspy.LM in case of an API

    adapter : dspy.adapter, default=None
        a dspy adapter e.g. dspy.adapters.ChatAdapter, dspy.adapters.JSONAdapter, defaults to LocalChatAdapter if not provided

    trace : bool, default=False
        In case adapter is None, passed on to LocalChatAdapter

    Returns
    -------

    An object of class localllm.dspy.LocalLLM in case lm starts with localllm/
    In all other cases, makes a connection to an LLM with dspy and returns a dspy.clients.lm.LM

    Examples
    --------

    >>> from localllm.config import localllm_connect
    >>> path = localllm_download_model("gemma-3-270m-it-Q8_0", overwrite=True)
    Downloading...
    >>> ################################################################################
    >>> ## Connect to an LLM with llama_cpp_python
    >>> ## 
    >>> llm = localllm_connect("localllm/gemma-3-270m-it-Q8_0")    
    >>> type(llm)
    <class 'localllm.dspy.LocalLLM'>
    >>>
    >>> ################################################################################    
    >>> ## Connect to an LLM with dspy - e.g. to connect to lmstudio or another model
    >>> ##    
    >>> config = dict(api_base = "http://localhost:1234/v1", api_key = "none", model_type = "chat", provider = "openai", cache = False, response_format = dict(type = "text"))
    >>> llm = localllm_connect("openai/translategemma-4b-it", model_kwargs = config)
    >>> type(llm)
    <class 'dspy.clients.lm.LM'>
    """

    if isinstance(lm, str):
        if lm.startswith("localllm/"):
            model_name = lm.replace("localllm/", "")
            model_name = localllm_download_model(model_name)
            ## Case where you specify localllm/gemma-3-4b-it-Q4_K_M for example
        else:
            model_name = lm
        if adapter is None:
            adapter = LocalChatAdapter(trace=trace)
        if os.path.exists(model_name):
            ## Local gguf LLM model
            from llama_cpp import Llama
            transformer = Llama(model_path=model_name, **model_kwargs)
            lm = LocalLLM(transformer)
            dspy.configure(lm=lm, adapter=adapter)
        else:
            ## Other API service providers
            lm = dspy.LM(model_name, **model_kwargs)
            dspy.configure(lm=lm, adapter=adapter)
        return lm
    else:
        raise NotImplementedError("lm can only be of type str for now")
