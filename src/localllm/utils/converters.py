from typing import Union
from s3generics import generic
import pandas as pd
import dspy

@generic
def tif(obj, *args, **kwargs):
    """Text Interchange Formats"""
    raise NotImplementedError(f"No tif() method for {type(obj).__name__}.")

@tif.register(pd.DataFrame)
def _tif_df(obj, docid_field = "doc_id", text_field = "text", target_field: Union[str | None] = None):
    if target_field is not None:
        if target_field not in obj.columns:
            raise Exception("target_field not in obj")
        else:    
            x = dict(doc_id = obj[docid_field], text = obj[text_field], target = obj[target_field])
    else:
        x = dict(doc_id = obj[docid_field], text = obj[text_field])
    x = pd.DataFrame(x)
    return x


@generic
def convert_dspy_example(obj, *args, **kwargs):
    """Convert to dspy examples"""
    raise NotImplementedError(f"No convert_dspy_example() method for {type(obj).__name__}.")

@convert_dspy_example.register(pd.DataFrame)
def _convert_dspy_example_df(obj, type = "classification", labels = None):
    raw_data = []
    if type == "classification":
        for i in range(len(obj)):
            text = obj["text"].iloc[i]
            target = obj["target"].iloc[i]
            if text is not None and target is not None:
                x = dict(text = text, target = target)
                ex = dspy.Example(x).with_inputs("text")
                raw_data.append(ex)
    elif type == "multilabel":  
        if labels is None:
            raise Exception("labels should be a list for type 'multilabel'")          
        for i in range(len(obj)):
            text = obj["text"].iloc[i]
            target = obj["target"].iloc[i]
            if text is not None and target is not None:
                x = dict(text = text)
                for item in labels:
                    default = []
                    empty = dict()
                    if item in target.keys():
                        empty[item] = target.get(item)
                    else:    
                        empty[item] = default
                    x.update(empty)                    
                ex = dspy.Example(x).with_inputs("text")
                raw_data.append(ex)
    else:
        raise NotImplementedError
    return raw_data