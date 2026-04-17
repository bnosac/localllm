# Changelog

## 0.3.0

* import localllm_connect as connect making sure you can do 
      import localllm
      localllm.connect("localllm/modelxyz")
* Move converters to localllm.utils.converters

## 0.2.2

* Added optional converters
    * from localllm.converters import tif, convert_dspy_example
    * added dependency on s3generics
    * added optional dependency on pandas for the converters
* Bump dependency to Python >=3.10
* LocalChatAdapter: more lenient parsing of the output fields [## value# ] and variants
* localllm_download_model, added:
    * gemma-4-E4B-it-Q8_0
    * gemma-4-E4B-it-Q4_K_M
    * translategemma-12b-it-Q4_K_M
    * gemma-3-4b-it-qat-Q4_0
* localllm_download_model, renamed 
    * gemma-4-e2b-it-Q8_0 to gemma-4-E2B-it-Q8_0
* localllm_download_model - reorder by bigger to smaller
* Add localllm_list_models to list up the models which you can download easily

## 0.2.1

* Exported localllm_connect
* localllm_download_model, added:
    * Qwen3.5-9B-Q4_K_M
    * Qwen3.5-4B-Q4_K_M
    * Qwen3.5-2B-Q4_K_M
    * Qwen3.5-0.8B-Q8_0
    * LFM2.5-350M-Q8_0
    * gemma-4-e2b-it-Q8_0
    * gemma-4-E2B-it-Q4_K_M
* LocalLLM
    * put __call__ in try-catch block and return [None] in case of an error (e.g. context to big for the model) + log the error in case of trace
    * in __call__ trace more arguments if requested

## 0.2.0

* Added LocalChatAdapter (extending BAMLAdapator) to parse out the LLM response which are wrongly formatted more gracefully defaulting to dspy_signature_defaults in case of complete failure
* Added dspy_signature_defaults
* localllm_download_model, added:
    * gemma-3-12b-it-qat-Q4_0
    * Qwen3-8B-Q4_K_M
    * Qwen3-4B-Instruct-Q4_K_M
    * LFM2.5-1.2B-Instruct-Q4_K_M
    * LFM2.5-1.2B-Instruct-Q8_0
* Added localllm.config.localllm_connect
* Bump dependencies to llama-cpp-python>=0.3.8 and dspy>=3.0.0

## 0.1.0

* Added functionality to download a model: localllm_download_model
* Added functionality run a local model (with llama-cpp) alongside dspy: LocalLLM
* localllm_download_model:
    * gemma-3-270m-it-Q8_0
    * gemma-3-270m-it-qat-Q4_0
    * gemma-3-1b-it-Q8_0
    * gemma-3-4b-it-Q4_K_M
    * GLM-4.6V-Flash-Q4_K_M
    * translategemma-4b-it-q8_0
