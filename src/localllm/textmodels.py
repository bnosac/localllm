import dspy
import random
import pandas as pd
from s3generics import predict, summary, coef
from typing import Iterable, Literal, Optional, Union
from collections.abc import Callable
from .utilities.train_test import train_test_split
from .utilities.converters import tif, convert_dspy_example

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
        self._optimized = None
        self.optimizer = None
    def __repr__(self) -> str:
        if self._optimized is None:
            return f"TextModelGEPA(budget='{self.auto}') [not fitted]"
        return (
            f"TextModelGEPA(budget='{self.auto}', classes={self.classnames}) "
            f"[fitted; n_train={self.n_train}, method={self.algorithm}]"
        )


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
        auto: Literal["light", "medium", "heavy"] = "light",
        train_size: int = 0.75,
        test_size: Union[float, int] = 0.1,
        seed: int = 4321,
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
    auto : {'light', 'medium', 'heavy'}
        GEPA ``auto`` budget. Default ``'light'``.        
    train_size : float
        Size of the training set. Defaults to the 0.75, indicating 75% of (the size of the data x - test_size )
    test_size : int
        Number of holdout test samples. Maximum number of training examples passed to the optimizer.
    seed : int
        Seed for shuffing the input data
    **gepa_kwargs
        Forwarded to ``dspy.GEPA``.

    Returns
    -------
    TextModelGEPA
        A GEPA prompt-finetuned classifier

    Examples
    --------

    >>> ######################################################################################
    >>> ## Define the model
    >>> ##
    >>> import localllm
    >>> from localllm import textmodel_gepa_classify
    >>> lm = localllm.connect("localllm/LFM2.5-1.2B-Instruct-Q8_0")
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
    >>> import localllm
    >>> from localllm import textmodel_gepa_classify
    >>> lm = localllm.connect("localllm/LFM2.5-1.2B-Instruct-Q8_0")
    >>> model = textmodel_gepa_classify(x = dataset["text"], y = dataset["target"], auto = None, train_size = 0.75)    
    >>> 
    >>> ######################################################################################
    >>> ## Define the model and auto-tune the prompt
    >>> ##    
    >>> config = dict(api_base = "http://localhost:1234/v1", api_key = "none", model_type = "chat", provider = "openai", cache = True, response_format = dict(type = "text"))
    >>> reflection_lm = localllm.connect("openai/gemma-4-E4b-it-GGUF", model_kwargs = config)        
    """
    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("You must first have configured dspy with either dspy.configure or localllm_connect")    
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
    train, testset = train_test_split(examples, test_size = test_size, random_state=seed)
    trainset, valset = train_test_split(train, train_size = train_size, random_state=seed)
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
    classify.set_lm(lm)
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
    optimizer = dspy.GEPA(metric = metric, reflection_lm = model.reflection_lm, auto = model.auto, **gepa_kwargs)
    model.program = optimizer.compile(model.module, trainset=trainset, valset=valset)
    model.optimizer = optimizer
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
    predictor = model.program.predict
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
    if hasattr(model.optimizer, "stats") and model.optimizer.stats:
        print(f"\n  Optimizer stats:")
        for k, v in model.optimizer.stats.items():
            print(f"    {k}: {v}")
    print(f"\n  Optimized components:")
    coef_df = coef(model)
    print(coef_df.to_string(index=False))