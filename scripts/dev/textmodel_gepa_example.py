import localllm
model_name = "localllm/gemma-4-E2B-it-Q8_0"
model_name = "localllm/Qwen3-8B-Q4_K_M"
model_name = "localllm/gemma-3-1b-it-Q8_0"
model_name = "localllm/LFM2.5-1.2B-Instruct-Q8_0"
model_name = "localllm/gemma-3-4b-it-Q4_K_M"
llm = localllm.connect(model_name, 
                       model_kwargs = dict(
                            verbose = False, 
                            n_ctx = 4096, n_gpu_layers = 0, n_threads = 1, seed = 4321),
                       adapter = LocalChatAdapter(trace = False),
                       trace = False) 
localllm.dspy._LOCAL_LLM_MODELS_LOADED

import localllm
from localllm import textmodel_gepa_classify, LocalChatAdapter    
from s3generics import summary, predict
#opts = dict(n_ctx = 4096, n_gpu_layers = 0, n_threads = 1, flash_attn = True, verbose = False, chat_format = None)
#lm = localllm.connect("localllm/LFM2.5-1.2B-Instruct-Q8_0", opts)
opts = dict(api_base = "http://localhost:1234/v1", api_key = "none", model_type = "chat", provider = "openai", cache = True, response_format = dict(type = "text"))    
reflect_lm = localllm.connect("openai/gemma-4-e4b-it", opts) 
lm = localllm.connect("openai/gemma-4-e2b-it", opts) 

######################################################################################
## Get data, define target to predict
##
import pandas as pd    
from localllm.data import data_be_parliament
from localllm.utilities.converters import tif
be = data_be_parliament()
be = pd.DataFrame.from_records(be)
be = be[be["question_theme_main"].isin(["VERVOERBELEID", "OPENBARE VEILIGHEID"])]
be = tif(be, docid_field = "doc_id", text_field = "question", target_field = "question_theme_main")
list(be.columns)

dataset = be.sample(200)

######################################################################################
## Define the model to use
## 
model = textmodel_gepa_classify(
    x = dataset["text"], 
    y = dataset["target"], 
    auto = "light", #max_metric_calls = 3, 
    reflection_minibatch_size = 5, reflection_lm = reflect_lm, test_size = 20, trace = True)
model = textmodel_gepa_classify(x = dataset["text"], y = dataset["target"], auto = None, max_metric_calls = 3, test_size = 3, num_threads = 1, trace = True)
model = textmodel_gepa_classify(x = dataset["text"], y = dataset["target"], auto = "light", max_metric_calls = 3, test_size = 3, num_threads = 1, trace=True)
model.eval_baseline
model.eval_tuned
scores = predict(model, newdata = ["We gaan met de trein op reis naar Blankenberge", "De politie is met man en macht op straat"])

max(model.program.detailed_results.val_aggregate_scores)

import re
be["question_themes"] = be["question_theme"].map(lambda x: [el.strip() for el in re.split(pattern = r"\|", string = x)])