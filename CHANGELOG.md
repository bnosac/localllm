# Changelog

## 0.2.1

* Exported localllm_connect
* localllm_download_model, added:
    * Qwen3.5-9B-Q4_K_M
    * Qwen3.5-4B-Q4_K_M
    * Qwen3.5-2B-Q4_K_M
    * Qwen3.5-0.8B-Q8_0

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
