import dspy
import json
import re
from typing import Any, Dict, get_origin, get_args
from dspy import ChatAdapter
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.utils.exceptions import AdapterParseError

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
    ...     answer:   str = dspy.OutputField(desc = "A city name")
    >>>
    >>> model = dspy.Predict(Go)
    >>> out = model(sentence="What is the capital of France")
    >>> out["answer"]
    'Paris'
    """

    def __init__(self, object=None, model="LocalLLM", model_type="chat", temperature=0.0, max_tokens=1000, cache=True, trace=True, **kwargs):
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
        call_kwargs = {**self.kwargs, **kwargs}
        if self.trace > 1:
            print("====================== START (LocalLLM.__call__) ======================")
            print(list(kwargs.keys()))        
            print(list(call_kwargs.keys()))
        try:
            response = self.llm.create_chat_completion_openai_v1(messages=messages, **call_kwargs)
            # Return in the format DSPy expects: list of strings
            out = [response.choices[0].message.content]
        except Exception as e: 
            if self.trace:
                print(e)
            out = [None]
        if self.trace > 1:
            print("====================== DONE (LocalLLM.__call__) ======================")        
        return out

    def forward(self, prompt=None, messages=None, **kwargs):
        if self.trace > 1:
            print("====================== START (LocalLLM.__forward__) ======================")
        """Forward method for regular DSPy operations"""
        if prompt is not None and messages is None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either prompt or messages must be provided")
        call_kwargs = {**self.kwargs, **kwargs}
        raw = self.llm.create_chat_completion_openai_v1(messages=messages, **call_kwargs)
        return raw

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        ## self.provider.kill(self, launch_kwargs)
        pass
        





class LocalChatAdapter(ChatAdapter):
    """
    A DSPy adapter that attempts progressively more lenient parsing strategies when strict JSON parsing fails. 
    Falls back to empty default values if all parsing strategies fail.

    Parameters
    ----------

    return_defaults_on_failure : bool, default=True
        If True, returns default empty values when all parsing strategies fail.
        If False, raises the original parsing error.
    
    trace : bool, default=False
        If True, prints debug information during parsing attempts.
    
    Returns
    --------

    LocalChatAdapter
        An object of type LocalChatAdapter, extending ChatAdapter

    Examples
    --------

    >>> import dspy
    >>> from llama_cpp import Llama
    >>> from localllm import LocalLLM, localllm_download_model, LocalChatAdapter
    >>> path = localllm_download_model("gemma-3-270m-it-Q8_0", trace = False)
    >>> transformer = Llama(model_path=path, n_gpu_layers=-1, flash_attn = False, n_ctx = 32768, n_threads = 1, seed = 4321, verbose = False)
    >>> dspy.configure(lm = LocalLLM(transformer, trace = False), adapter = LocalChatAdapter(return_defaults_on_failure = True, trace = False))
    >>> class Go(dspy.Signature):
    ...     sentence: str = dspy.InputField(desc = "A question")
    ...     answer: str = dspy.OutputField(desc="A city name")
    >>>    
    >>> model = dspy.Predict(Go)
    >>> out = model(sentence="What city is the Capital of Belgium")
    """
    
    def __init__(self, return_defaults_on_failure: bool = True, trace: bool = False):
        super().__init__()
        self.return_defaults_on_failure = return_defaults_on_failure
        self.trace = trace

    def parse(self, signature, completion: str) -> Dict[str, Any]:
        """
        Override parse method with fallback strategies for malformed JSON.
        """
        # Try the standard ChatAdapter parsing first
        if self.trace > 1:
            print("====================== Parsing LLM completion ======================")
            print(completion)
        try:            
            return super().parse(signature, completion)
        except (AdapterParseError, json.JSONDecodeError, Exception) as e:
            # If strict parsing fails, try progressive fallback strategies            
            try:
                if self.trace:
                    print("====================== LocalChatAdapter LLM did not provide valid structured output, fallback to lenient parsing strategy ======================")
                    if self.trace <= 1:
                        print(completion)
                return self._lenient_parse(signature, completion, original_error=e)
            except Exception as lenient_error:
                if self.trace:
                    print("Could not leniently parse the LLM response.")
                # If all strategies fail, return default empty values
                if self.return_defaults_on_failure:
                    return dspy_signature_defaults(signature)
                else:
                    # Raise the original error if available, otherwise the lenient error
                    raise e if isinstance(e, AdapterParseError) else lenient_error
                
    
    def _lenient_parse(self, signature, completion: str, original_error=None) -> Dict[str, Any]:
        """
        Implement progressive fallback parsing strategies.
        """
        text = completion.strip()
        
        # Strategy 1: Try to fix common JSON issues
        try:
            cleaned = self._clean_json_response(text)
            parsed = json.loads(cleaned)
            return self._validate_and_extract(signature, parsed)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks or mixed content
        try:
            extracted = self._extract_json_from_text(text)
            if extracted:
                parsed = json.loads(extracted)
                return self._validate_and_extract(signature, parsed)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Strategy 3: Try to complete incomplete JSON
        try:
            completed = self._complete_incomplete_json(text)
            parsed = json.loads(completed)
            return self._validate_and_extract(signature, parsed)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Strategy 4: Extract fields using regex as last resort
        try:
            return self._parse_dspy_markers(signature, text)
        except Exception:
            pass
        
        # Strategy 5: Extract fields using regex as last resort
        try:
            return self._regex_field_extraction(signature, text)
        except Exception:
            pass
        
        # All strategies failed - raise to trigger default values
        raise ValueError(f"All lenient parsing strategies failed")
        
    def _clean_json_response(self, text: str) -> str:
        """
        Clean common JSON formatting issues.
        """
        # Remove markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        
        # Remove any leading/trailing text before/after JSON
        # Try to find the outermost { } or [ ]
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return text
        
        if start_brace == -1:
            start = start_bracket
            end_char = ']'
        elif start_bracket == -1:
            start = start_brace
            end_char = '}'
        else:
            start = min(start_brace, start_bracket)
            end_char = '}' if start == start_brace else ']'
        
        # Find matching closing brace/bracket
        end = text.rfind(end_char)
        if end != -1:
            text = text[start:end+1]
        
        return text.strip()
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text that may contain other content.
        """
        # Try to find JSON within markdown code blocks
        json_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find raw JSON object or array
        # Look for the first { and last } or first [ and last ]
        json_obj_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        match = re.search(json_obj_pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        
        json_arr_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'
        match = re.search(json_arr_pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        
        return text
    
    def _complete_incomplete_json(self, text: str) -> str:
        """
        Attempt to complete truncated JSON (like your example).
        """
        text = self._clean_json_response(text)
        
        # Remove trailing commas before closing brackets/braces
        text = re.sub(r',(\s*[\]}])', r'\1', text)
        
        # Count braces and brackets to detect incomplete structures
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # Check if we're in the middle of a string value
        # Count quotes, if odd number, we need to close the string
        # More sophisticated: count non-escaped quotes
        in_string = False
        escape_next = False
        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
        
        if in_string:
            text += '"'
        
        # Add missing closing characters
        if open_brackets > 0:
            text += ']' * open_brackets
        if open_braces > 0:
            text += '}' * open_braces
        
        return text
    
    def _validate_and_extract(self, signature, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parsed JSON has required fields from signature.
        Fill missing fields with default values.
        """
        # Extract expected output fields from signature
        output_fields = {}
        for name, field in signature.fields.items():
            if hasattr(field, 'json_schema_extra') and field.json_schema_extra:
                if field.json_schema_extra.get("__dspy_field_type") == "output":
                    output_fields[name] = field
        
        # If no output fields defined, return as-is
        if not output_fields:
            return parsed
        
        result = {}
        for field_name, field in output_fields.items():
            if field_name in parsed:
                result[field_name] = parsed[field_name]
            else:
                # Field missing - try case-insensitive match
                found = False
                for key in parsed.keys():
                    if key.lower() == field_name.lower():
                        result[field_name] = parsed[key]
                        found = True
                        break
                
                # If still not found, use default value
                if not found:
                    result[field_name] = get_default_for_type(field.annotation)
        
        return result
    
    def _parse_dspy_markers(self, signature, text: str) -> Dict[str, Any]:
        """
        Parse DSPy's [[ ## field ## ]] marker format.
        Tolerates malformed variants like [[ ## field ##] (single closing bracket).
        """
        output_fields = {
            name: field
            for name, field in signature.fields.items()
            if hasattr(field, 'json_schema_extra') and field.json_schema_extra and field.json_schema_extra.get("__dspy_field_type") == "output"
        }
        if not output_fields:
            return {}
        # Matches on:
        #   [[ ## field ## ]]   (well-formed)
        #   [[ ## field ##]     (single closing bracket, no space)
        #   [[ ## field ##]]    (no space before closing)
        #   [ # field##]]       (and all variants of these)
        # marker_pattern = re.compile(
        #     r'\[\[\s*##\s*(\w+)\s*##\s*\]{1,2}\]?',
        #     re.IGNORECASE
        # )
        marker_pattern = re.compile(
            r'\[{1,3}\s*#{1,3}\s*(\w+)\s*#{1,3}\s*\]{1,3}',
            re.IGNORECASE
        )
        # Split text into (field_name, content) segments
        segments = []
        last_end = 0
        last_field = None
        for match in marker_pattern.finditer(text):
            if last_field is not None:
                segments.append((last_field, text[last_end:match.start()].strip()))
            last_field = match.group(1).strip()
            last_end = match.end()
        # Capture the final segment after the last marker
        if last_field is not None:
            segments.append((last_field, text[last_end:].strip()))
        if not segments:
            return {}
        result = {}
        for field_name, value in segments:
            # Normalize field name lookup (case-insensitive)
            matched_field = next(
                (f for f in output_fields if f.lower() == field_name.lower()),
                None
            )
            if matched_field:
                result[matched_field] = value
        # Fill defaults for any missing output fields
        for field_name, field in output_fields.items():
            if field_name not in result:
                result[field_name] = get_default_for_type(field.annotation)
        return result
    
    def _regex_field_extraction(self, signature, text: str) -> Dict[str, Any]:
        """
        Last resort: extract fields using regex patterns.
        Returns partial results with defaults for missing fields.
        """
        output_fields = {}
        for name, field in signature.fields.items():
            if hasattr(field, 'json_schema_extra') and field.json_schema_extra:
                if field.json_schema_extra.get("__dspy_field_type") == "output":
                    output_fields[name] = field
        
        result = {}
        for field_name, field in output_fields.items():
            # Try to find field: value patterns (with or without quotes on field name)
            patterns = [
                rf'"{field_name}"\s*:\s*(\[[^\]]*\]|"[^"]*"|\d+\.?\d*|true|false|null)',
                rf'{field_name}\s*:\s*(\[[^\]]*\]|"[^"]*"|\d+\.?\d*|true|false|null)',
            ]
            found = False
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    value_str = match.group(1).strip()
                    try:
                        # Try to parse as JSON
                        result[field_name] = json.loads(value_str)
                        found = True
                    except json.JSONDecodeError:
                        # If that fails, treat as string (remove quotes if present)
                        result[field_name] = value_str.strip('"')
                        found = True
                    break
            
            # If field not found, use default value
            if not found:
                result[field_name] = get_default_for_type(field.annotation)
        
        return result





def dspy_signature_defaults(signature: dspy.Signature) -> Dict[str, Any]:
    """
    Generate default empty values for all output fields in a DSPy signature.
    
    This function extracts all output fields from a DSPy signature and creates
    a dictionary with appropriate default empty values for each field based on
    its type annotation.
    
    Parameters
    ----------
    signature : dspy.Signature
        A DSPy signature object containing field definitions with type annotations
    
    Returns
    -------
    Dict[str, Any]
        A dictionary mapping field names to their default empty values.
        Only includes fields marked as output fields.
    
    Examples
    --------
    
    >>> import dspy
    >>> from localllm import dspy_signature_defaults
    >>> class MySignature(dspy.Signature):
    ...     question: str = dspy.InputField()
    ...     answer: str = dspy.OutputField()
    ...     names: list[str] = dspy.OutputField()
    ...     count: int = dspy.OutputField()
    >>> defaults = dspy_signature_defaults(MySignature)
    >>> defaults
    {'answer': None, 'names': [], 'count': None}
    
    Notes:
    ------
    - Only processes fields with __dspy_field_type == "output"
    - Input fields are excluded from the result
    - Uses get_default_for_type() to determine appropriate defaults
    """
    result = {}    
    for name, field in signature.fields.items():
        # Check if this is an output field
        if hasattr(field, 'json_schema_extra') and field.json_schema_extra:
            if field.json_schema_extra.get("__dspy_field_type") == "output":
                result[name] = get_default_for_type(field.annotation)    
    return result



def get_default_for_type(type_annotation) -> Any:
    """
    Get appropriate default empty value based on type annotation.
    
    This function provides sensible default values for common Python types,
    which is useful when parsing fails and you need to return a valid structure.
    
    Parameters:
    -----------
    type_annotation : type
        A Python type annotation (e.g., str, int, list[str], Optional[int])
    
    Returns:
    --------
    Any
        An appropriate default empty value for the given type:
        - list → []
        - dict → {}
        - set → set()
        - tuple → ()
        - str → None
        - int → None
        - float → None
        - bool → False
        - Optional[X] → default for X or None
        - Union[X, Y, ...] → default for first non-None type
        - Unknown types → None
    
    Examples:
    ---------
    >>> get_default_for_type(list)
    []
    >>> get_default_for_type(str)
    >>> get_default_for_type(int)
    >>> from typing import Optional
    >>> get_default_for_type(Optional[str])
    >>> get_default_for_type(list[int])
    []
    """
    # Handle None type
    if type_annotation is type(None):
        return None
    
    # Get the origin type for generics (e.g., list, dict)
    origin = get_origin(type_annotation)
    
    # Handle common types
    if origin is list or type_annotation is list:
        return []
    elif origin is dict or type_annotation is dict:
        return {}
    elif origin is set or type_annotation is set:
        return set()
    elif origin is tuple or type_annotation is tuple:
        return ()
    elif type_annotation is str:
        return None
    elif type_annotation is int:
        return None
    elif type_annotation is float:
        return None
    elif type_annotation is bool:
        return False
    
    # Handle Optional types (Union[X, None])
    if origin is type(None) or (hasattr(type_annotation, '__origin__') and 
                                 type_annotation.__origin__ is type(None)):
        return None
    
    # Try to handle Union types
    try:
        from typing import Union
        if origin is Union:
            args = get_args(type_annotation)
            # Return default for the first non-None type
            for arg in args:
                if arg is not type(None):
                    return get_default_for_type(arg)
            return None
    except:
        pass
    
    # For unknown types, try common defaults
    if hasattr(type_annotation, '__origin__'):
        origin = type_annotation.__origin__
        if origin is list:
            return []
        elif origin is dict:
            return {}
    
    # Default fallback
    return None
