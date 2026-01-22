from typing import Any
import dspy


class LocalLLM(dspy.BaseLM):
    """
    Create a Local LLM object which you can use alongside dspy

    Parameters
    --------

    object (llama_cpp.llama.Llama) :
        An object of class 'llama_cpp.llama.Llama' as returned by Llama from the llama-cpp-python package
    model (str) :
        A name that you provide to the model. Defaults to 'LocalLLM'
    model_type (str) :
        String with the type of model. e.g. 'chat', 'responses'. Currently only tested with type: 'chat'
    temperature (float) :
        The model temperature to use. Defaults to 0
    max_tokens (int) :
        Maximum number of tokens. Defaults to 1000
    cache (bool) :
        Use the cache to avoid recomputing the same llm call twice. Defaults to True.
    trace (bool) :
        Boolean indicating to log internal calls. Defaults to False.

    Returns
    --------

    dspy.BaseLM
        An object of type dspy.BaseLM

    Examples
    --------

    >>> import dspy
    >>> from llama_cpp import Llama
    >>> from localllm import LocalLLM, localllm_download_model
    >>> path = localllm_download_model("gemma-3-270m-it-Q8_0", overwrite = True, trace = False)
    >>> transformer = Llama(model_path=path, n_gpu_layers=-1, flash_attn = False, n_ctx = 32768, n_threads = 1, seed = 4321, verbose = False)
    >>> out = transformer("How much is 4x4")
    >>>
    >>> dspy.configure(lm = LocalLLM(transformer))
    >>> class Go(dspy.Signature):
    ...     sentence: str = dspy.InputField(desc = "A question")
    ...     value = dspy.OutputField(desc="A city")
    >>>
    >>> model = dspy.Predict(Go)
    >>> out = model(sentence="What is the capital of France")
    >>> out["value"]
    'Paris'
    """

    def __init__(self, object=None, model="LocalLLM", model_type="chat", temperature=0.0, max_tokens=1000, cache=True, trace=False, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.history: list[dict[str, Any]] = []
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.llm = object
        self.trace = trace

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Handle both string prompts and message lists for GEPA compatibility"""
        if prompt is not None and messages is None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if self.trace:
            print("====================== START (__call__) ======================")
            print(list(kwargs.keys()))
        call_kwargs = {**self.kwargs, **kwargs}
        response = self.llm.create_chat_completion_openai_v1(messages=messages, **call_kwargs)
        if self.trace:
            print("====================== DONE (__call__) ======================")
        # Return in the format DSPy expects: list of strings
        return [response.choices[0].message.content]

    def forward(self, prompt=None, messages=None, **kwargs):
        if self.trace:
            print("====================== START (__forward__) ======================")
        """Forward method for regular DSPy operations"""
        if prompt is not None and messages is None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either prompt or messages must be provided")
        call_kwargs = {**self.kwargs, **kwargs}
        raw = self.llm.create_chat_completion_openai_v1(messages=messages, **call_kwargs)
        return raw
