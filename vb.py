
from localllm import textmodel_gepa_classify
from localllm.converters import tif, convert_dspy_example
from localllm.data import data_be_parliament
import pandas as pd
import dspy
from localllm import localllm_connect
model_name = "localllm/gemma-3-4b-it-Q4_K_M"
model_name = "localllm/Qwen3-4B-Instruct-Q4_K_M"
model_name = "localllm/LFM2.5-1.2B-Instruct-Q8_0"
# llm = localllm_connect(model_name, 
#                        model_kwargs = dict(
#                             verbose = False, 
#                             n_ctx = 32768, n_gpu_layers = -1, n_threads = 1, flash_attn = True, swa_full = True, seed = 4321)) 

from localllm.converters import tif, convert_dspy_example
from localllm.data import data_be_parliament
import pandas as pd
import dspy
from localllm import localllm_connect, LocalChatAdapter
config = dict(api_base = "http://localhost:1234/v1", api_key = "none", 
              model_type = "chat", provider = "openai", 
              cache = True, response_format = dict(type = "text"))
config = dict(api_base = "http://localhost:1234/v1", api_key = "none", 
              model_type = "chat", provider = "openai", 
              cache = True, response_format=None)

#lm = localllm_connect("openai/qwen3.5-9b", model_kwargs = config)
lm = localllm_connect("openai/lfm2-1.2b", model_kwargs = config, adapter = LocalChatAdapter(trace = 2))
lm = localllm_connect("openai/liquid/lfm2.5-1.2b", model_kwargs = config, adapter = LocalChatAdapter(trace = 2))
lm = localllm_connect("openai/gemma-3-270m-it", model_kwargs = config, adapter = LocalChatAdapter(trace = 2))
lm = localllm_connect("openai/gemma-4-E2B-it-GGUF", model_kwargs = config, adapter = LocalChatAdapter(trace = 1))
## Order is important
reflection_lm = localllm_connect("openai/gemma-4-E4b-it-GGUF", model_kwargs = config, adapter = LocalChatAdapter(trace = 2))
lm = localllm_connect("openai/gemma-4-E2B-it-GGUF", model_kwargs = config, adapter = LocalChatAdapter(trace = 2))

dspy.settings.lm.model
reflection_lm
#dspy.disable_logging() 
#dspy.disable_litellm_logging()

x = data_be_parliament()
be = pd.DataFrame.from_records(x)
be["question_theme_main"].value_counts()
be["is_VERVOERBELEID"] = be["question_theme_main"] == "VERVOERBELEID"
be = be[be["question_theme_main"].isin(["VERVOERBELEID", "OPENBARE VEILIGHEID"])]
be = tif(be, docid_field = "doc_id", text_field = "question", target_field = "question_theme_main")
#be = tif(be, docid_field = "doc_id", text_field = "question", target_field = "language")
be["target"].value_counts()
#d = be.sample(1000, weights='target')
d = be.sample(100)
d["target"].value_counts()

model = textmodel_gepa_classify(d, type = "ChainOfThought", reflection_lm = reflection_lm, track_stats = True)

class GEPA_Classify(dspy.Signature):
  "Classify the text with either one of the following categories: dutch or french."
  text:   str = dspy.InputField(desc = "A question asked in Belgium Parliament")
  target: Literal["dutch", "french"] = dspy.OutputField(desc = "The language of the text, either dutch or french")

model = textmodel_gepa_classify(d, module = GEPA_Classify, type = "ChainOfThought", reflection_lm = reflection_lm)
model = textmodel_gepa_classify(d, module = GEPA_Classify, type = "ChainOfThought", add_format_failure_as_feedback = False)
summary(model)
print(model.program.signature.instructions)
print(model.program.detailed_results)

def metric(gold, prediction, trace=None, pred_name=None, pred_trace=None):
    total = float(gold.target == prediction.target)
    return total

len(model._data["validation"])
import dspy
evaluate = dspy.Evaluate(
    devset=model._data["test"], metric=metric,
    num_threads=2,
    display_table=True,
    display_progress=True
)
scores = evaluate(model.program)

pd.DataFrame.from_records(scores.toDict())