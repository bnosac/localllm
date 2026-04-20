import dspy
import random
import pandas as pd
from s3generics import predict, summary, coef
from typing import Iterable, Literal, Optional, Union
from collections.abc import Callable
from .utilities.train_test import train_test_split
from .utilities.converters import convert_dspy_example

class TextModelGEPA:
    """
    GEPA-optimized DSPy text classifier.
    """
    def __init__(
        self,
        lm: dspy.LM,
        reflection_lm: dspy.LM,
        module = None,        
        n_train: int = None,
        n_test: int = None,
        n_validation: int = None,
        auto: Literal["light", "medium", "heavy"] = "light",
        classnames: Optional[list[str]] = None,
        seed: int = 0,
        gepa_kwargs: Optional[dict] = None,
    ) -> None:
        self.n_train = n_train
        self.n_test = n_test
        self.n_validation = n_validation
        self._data = None
        self.auto = auto
        self.seed = seed        
        self.classnames: Optional[list[str]] = classnames
        self.module = module        
        self.lm = lm
        self.reflection_lm = reflection_lm
        self.gepa_kwargs: dict = gepa_kwargs or {}        
        self.algorithm: Optional[str] = None
        self.program = None
        self.optimizer = None
        self.eval_baseline = None
        self.eval_tuned = None
    def __repr__(self) -> str:
        if self.program is None:
            out = f"TextModelGEPA(budget='{self.auto}') [not fitted]"
        else:
            out = (
                f"TextModelGEPA(budget='{self.auto}', classes={self.classnames}) "
                f"[fitted; n_train={self.n_train}, method={self.algorithm}]"
            )
        return out


def eval_classification(example, prediction, trace=None) -> bool:
    return example.target == prediction.target

def eval_classification_with_feedback(gold, prediction, trace=None, pred_name=None, pred_trace=None):
    total = float(gold.target == prediction.target)
    return dspy.Prediction(score=total, feedback=None)


def textmodel_gepa_classify(        
        x: Union[pd.DataFrame, Iterable[str]] = None,
        y: Optional[Iterable[str]] = None,
        module: dspy.Signature = None,
        which: Literal["predict", "chainofthought"] = "predict",        
        reflection_lm: Optional[dspy.LM] = None,        
        metric: Union[Callable, str] = "accuracy",
        auto: Literal["light", "medium", "heavy", None] = "light",
        train_size: Union[float, int] = 0.75,
        test_size: Union[float, int] = 0.1,
        seed: int = 4321,
        track_stats: bool = True,
        num_threads: int = 1,
        trace: bool = False,
        **gepa_kwargs
) -> TextModelGEPA:
    """
    Fit a GEPA-optimized text classification model using DSPy.

    Parameters
    ----------    
    x : Union[pd.DataFrame, Iterable[str]]
        If x is a pd.DataFrame, it should be a TIF-compatible DataFrame (with doc_id, text, target columns) 
        If x is a list of str, you need to supply *y*
    y : Optional[Iterable[str]] or None
        A list of str containing the labels to predict, required when *x* is raw text.
    module: dspy.Signature 
        A dspy signature to tune, with dspy.Predict or dspy.ChainOfThought where the signature has inputfield 'text' and outputfield 'target'
    which : {'predict', 'chainofthought'}
        Either Predict or ChainOfThought
    reflection_lm : dspy.LM or None
        Language model used by GEPA for reflection. Defaults to the configured dspy lm.
    metric : 'accuracy'
        Either a str with values 'accuracy' or a callable which returns how good the prediction was. Defaults to accuracy.
    auto : {'light', 'medium', 'heavy', None}
        GEPA ``auto`` budget. Default ``'light'``.        
    train_size : float
        Size of the training set. Defaults to the 0.75, indicating 75% of (the size of the data x - test_size )
    test_size : int
        Number of holdout test samples.
    seed : int
        Seed for shuffing the input data
    track_stats: bool
        Track stats on evaluation dataset
    num_threads: bool
        Number of threads to use in GEPA. Defaults to 1. Only change this if the default lm you are using is a remote lm served through an API.
    trace: bool
        Boolean allowing to disable dspy logging. Defaults to True.
    **gepa_kwargs
        Forwarded to ``dspy.GEPA``.

    Returns
    -------
    TextModelGEPA
        A GEPA prompt-finetuned classifier

    Examples
    --------

    >>> ######################################################################################
    >>> ## Define the model to use
    >>> ##
    >>> import localllm
    >>> from localllm import textmodel_gepa_classify    
    >>> ##
    >>> ## Example connecting to a local LLM running on LMStudio or an API running e.g. on your computer
    >>> ##
    >>> opts = dict(api_base = "http://localhost:1234/v1", api_key = "none", model_type = "chat", provider = "openai", cache = True, response_format = dict(type = "text"))
    >>> lm = localllm.connect("openai/gemma-4-E2B-it-GGUF", opts)                                                                                                              # doctest: +SKIP
    >>> ##
    >>> ## Example to connect to a local llm directly in Python
    >>> ##
    >>> m = localllm_download_model("Qwen3-4B-Instruct-Q4_K_M", overwrite=True, trace = False)
    >>> lm = localllm.connect("localllm/Qwen3-4B-Instruct-Q4_K_M")     
    >>> 
    >>> ######################################################################################
    >>> ## Get data, define target to predict
    >>> ##
    >>> import pandas as pd    
    >>> from localllm.data import data_be_parliament
    >>> from localllm.utilities.converters import tif
    >>> be = data_be_parliament()
    >>> be = pd.DataFrame.from_records(be)
    >>> be = be[be["question_theme_main"].isin(["VERVOERBELEID", "OPENBARE VEILIGHEID"])]
    >>> be = tif(be, docid_field = "doc_id", text_field = "question", target_field = "question_theme_main")
    >>> list(be.columns)
    ['doc_id', 'text', 'target']
    >>> dataset = be.sample(100)
    >>> 
    >>> ######################################################################################
    >>> ## Auto-tune the prompt using GEPA
    >>> ##
    >>> from s3generics import summary, predict
    >>> model = textmodel_gepa_classify(x = dataset["text"], y = dataset["target"], auto = None, max_metric_calls = 10, reflection_minibatch_size = 5)
    >>> score = predict(model, ["We gaan met de trein op reis naar Blankenberge", "De politie is met man en macht op straat"])
    >>> score  
    ['VERVOERBELEID', 'OPENBARE VEILIGHEID']
    >>> model = textmodel_gepa_classify(x = dataset["text"], y = dataset["target"], auto = "light")                                  # doctest: +SKIP
    >>> score = predict(model, ["We gaan met de trein op reis naar Blankenberge", "De politie is met man en macht op straat"])       # doctest: +SKIP
    >>> summary(model)                                                                                                               # doctest: +SKIP
    Method       : Classification (DSPy GEPA): predict
    GEPA auto    : None
    Classes      : ['OPENBARE VEILIGHEID', 'VERVOERBELEID']    
    ...
    >>> ######################################################################################
    >>> ## Add a better reflection model which is normally a better model
    >>> ##    
    >>> # reflection model
    >>> # opts = dict(api_base = "http://localhost:1234/v1", api_key = "none", model_type = "chat", provider = "openai", cache = True, response_format = dict(type = "text"))
    >>> # reflection_lm = localllm.connect("openai/gemma-4-E4b-it-GGUF", opts)    
    """
    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("You must first have configured dspy with either dspy.configure or localllm_connect")    
    if not trace:
        dspy.disable_logging() 
        dspy.disable_litellm_logging()   
    if isinstance(x, pd.DataFrame):
        tif_df = x
    else:
        tif_df = pd.DataFrame({
            "doc_id": range(len(list(x))), 
            "text": list(x), 
            "target": list(y)})
    tif_df = tif_df[~tif_df["target"].isna()]
    examples = convert_dspy_example(tif_df, type="classification")
    random.Random(seed).shuffle(examples)
    ## All target classes, sorted
    classes = sorted({e.target for e in examples})
    ## Split in holdout set of baseline evaluation / train / test
    if test_size > 0:
        train, testset = train_test_split(examples, test_size = test_size, random_state=seed)
        trainset, valset = train_test_split(train, train_size = train_size, random_state=seed)
    else:
        testset = []    
        trainset, valset = train_test_split(examples, train_size = train_size, random_state=seed)
    ## Define the module to tune
    if module is None:
        options = classes
        #options = ", ".join(options)
        class GEPA_Classify(dspy.Signature):
            f"""Classify the text with either one of the following categories: {options}."""
            text:   str = dspy.InputField()
            target: str = dspy.OutputField(desc=f"The classification, either one of: {options}")
        if which in ("predict", "Predict"):
            classify = dspy.Predict(GEPA_Classify)
        elif which in ("chainofthought", "ChainOfThought"):
            classify = dspy.ChainOfThought(GEPA_Classify)
        else: 
            raise ValueError("Provided which is not Predict/predict or ChainOfThought/chainofthought") 
    else:
        if which in ("predict", "Predict"):
            classify = dspy.Predict(module)
        elif which in ("chainofthought", "ChainOfThought"):
            classify = dspy.ChainOfThought(module)
        else:
            if module is None:
                raise ValueError("In module you must provide the output of dspy.ChainOfThought or dspy.Predict") 
            classify = module
    ## Define the module to tune
    model = TextModelGEPA(
        module = classify,
        lm = lm,
        reflection_lm = reflection_lm or lm,
        classnames = classes,
        auto = auto,
        n_train = len(trainset),
        n_validation = len(valset),
        n_test = len(testset),
        seed = seed,
        gepa_kwargs=gepa_kwargs,
    )
    model._data = dict(train = trainset, validation = valset, test = testset)    
    if metric == "accuracy":
        metric = eval_classification_with_feedback
    ## Baseline evaluation on holdout testset
    if test_size > 0:
        evaluate = dspy.Evaluate(devset=testset, metric=metric, display_progress=trace, num_threads = num_threads)
        testset_baseline = evaluate(model.module, num_threads = num_threads)   
        if trace:
            print("Evaluation on baseline: ", testset_baseline)
        model.eval_baseline = testset_baseline    
    ## Tune module
    optimizer = dspy.GEPA(metric = metric, reflection_lm = model.reflection_lm, auto = model.auto, track_stats = track_stats, num_threads = num_threads, **gepa_kwargs)    
    model.program = optimizer.compile(model.module, trainset=trainset, valset=valset)
    model.optimizer = optimizer    
    if test_size > 0:
        evaluate = dspy.Evaluate(devset=testset, metric=metric, display_progress=trace, num_threads = num_threads)
        testset_tuned = evaluate(model.program, num_threads = num_threads)        
        if trace:
            print("Evaluation on testset: ", testset_tuned)
        model.eval_tuned = testset_tuned    
    #max(model.program.detailed_results.val_aggregate_scores)
    model.algorithm = "Classification (DSPy GEPA): " + which
    return model


@predict.register(TextModelGEPA)
def _predict_textmodelgepa(
    model: TextModelGEPA,
    newdata: Union[Iterable[str]],
    **kwargs,
) -> list:
    """
    Predict class labels based on a GEPA-tuned classifier.
    """
    out = []
    for t in newdata:
        scores = model.program(text = t)
        scores = scores.target
        out.append(scores)
    return [model.program(text=t).target for t in newdata]


@coef.register(TextModelGEPA)
def _coef_textmodelgepa(model: TextModelGEPA, **kwargs) -> pd.DataFrame:
    """
    Return the GEPA-optimized and few-shot demonstrations.
    """
    predictor = model.program.predictors()[0]
    rows = []
    # Fine-tuned instructions
    if hasattr(predictor, "extended_signature"):
        rows.append({
            "component": "instruction", 
            "content": predictor.extended_signature.instructions})
    # Few-shot demonstrations
    for i, demo in enumerate(getattr(predictor, "demos", [])):
        rows.append({
            "component": f"demo_{i+1}",
            "content":   f"[text] {demo.get('text', '')} → [target] {demo.get('target', '')}",
        })
    out = pd.DataFrame(rows, columns=["component", "content"])
    return out


@summary.register(TextModelGEPA)
def _summary_textmodelgepa(model: TextModelGEPA, **kwargs) -> None:
    """Print a summary of the fitted GEPA model."""
    print(f"  Method       : {model.algorithm}")
    print(f"  GEPA auto    : {model.auto}")
    print(f"  Classes      : {model.classnames}")
    print(f"  Train set    : {model.n_train} examples (seed={model.seed})")
    print(f"  LM (infer)   : {model.lm}")
    print(f"  LM (reflect) : {model.reflection_lm}")
    if hasattr(model.program, "detailed_results") and model.optimizer.track_stats:
        print(f"  Optimizer stats:")
        print(f"    - Amount of time metric is called {model.program.detailed_results.total_metric_calls}")
        print(f"    - Best candidate:\n", model.program.detailed_results.best_candidate)
        print(f"  Candidate scores:")
        for i in range(len(model.program.detailed_results.candidates)):
            print("    - Validation set aggregate evaluation score: ", model.program.detailed_results.val_aggregate_scores[i])
            print(model.program.detailed_results.candidates[i])            
