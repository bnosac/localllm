from typing import List, Optional
from dataclasses import dataclass
# import re


@dataclass
class TextSpan:
    text: str
    start: int
    end: int

    def __repr__(self):
        return f"TextSpan(text='{self.text}', start={self.start}, end={self.end})"


def txt_locate(text: str, elements: List[str]) -> List[Optional[TextSpan]]:
    """
    Find start and end positions of text elements in a larger text.

    Parameters
    ----------

    text : str
        A text string to search for elements within it.

    elements : List[str]
        A list of strings to find in text

    Returns
    -------

    List[TextSpan]
        List of TextSpan objects with the first start/end positions of elements in text.
        The list is the same length as elements

    Examples
    --------

    >>> text = "Hello world! This is a test."
    >>> elements = ["world", "test", "notfound"]
    >>> loc = txt_locate(text, elements)
    >>> loc
    [TextSpan(text='world', start=6, end=11), TextSpan(text='test', start=23, end=27), None]

    """
    results = []
    for element in elements:
        start_pos = text.find(element)
        if start_pos != -1:
            end_pos = start_pos + len(element)
            results.append(TextSpan(text=element, start=start_pos, end=end_pos))
        else:
            results.append(None)
    return results


def txt_locate_all(text: str, elements: str | list[str]) -> List[TextSpan]:
    """
    Find all occurrences of a text element (in case it appears multiple times).

    Parameters
    ----------

    text : str
        A text string to search for elements within it.

    elements : str | list[str]
        A string to lookup in text or a list of strings to find in text

    Returns
    -------

    List[TextSpan]
        List of TextSpan objects with start/end positions


    Examples
    --------

    >>> text = "Hello world! This is a test find the a world in here."
    >>> loc = txt_locate_all(text, "world")
    >>> loc
    [TextSpan(text='world', start=6, end=11), TextSpan(text='world', start=39, end=44)]
    >>> loc = txt_locate_all(text, "notinthetext")
    >>> loc
    []
    >>> loc = txt_locate_all(text, ["test", "a", "world"])
    >>> loc
    [TextSpan(text='test', start=23, end=27), TextSpan(text='a', start=21, end=22), TextSpan(text='a', start=37, end=38), TextSpan(text='world', start=6, end=11), TextSpan(text='world', start=39, end=44)]

    """
    results = []
    start = 0
    if isinstance(elements, list):
        for element in elements:
            results.extend(txt_locate_all(text, element))
        return results
    element = elements
    while True:
        pos = text.find(element, start)
        if pos == -1:
            break
        results.append(TextSpan(text=element, start=pos, end=pos + len(element)))
        start = pos + 1
    return results


def merge_spans(text: str, spans: List[TextSpan], skip_spaces: bool = True) -> List[TextSpan]:
    """
    Combine TextSpan elements which are either overlapping or adjacent where only whitespace separates them.

    Parameters
    ----------

    text : str
        A text string where spans are extracted from

    spans: List[TextSpan]
        A list of TextSpan objects to look for overlaps and to combine

    skip_spaces: bool
        If True, combine spans which are not overlapping but which do have only whitespace between them

    Returns
    -------

    List[TextSpan]
        List of TextSpan objects with start/end positions

    Examples
    --------

    >>> text = "Hello world! This is a test find the world in here."
    >>> loc = txt_locate_all(text, "world")
    >>> combined = merge_spans(text, loc, skip_spaces=True)
    >>> loc = txt_locate_all(text, "world") + txt_locate_all(text, " wor")
    >>> loc
    [TextSpan(text='world', start=6, end=11), TextSpan(text='world', start=37, end=42), TextSpan(text=' wor', start=5, end=9), TextSpan(text=' wor', start=36, end=40)]
    >>> combined = merge_spans(text, loc, skip_spaces=True)
    >>> combined
    [TextSpan(text=' world', start=5, end=11), TextSpan(text=' world', start=36, end=42)]
    """
    result = []
    size = len(spans)
    if not spans or size == 0:
        return
    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda x: x.start)
    # Loop over all spans and combine if overlapping or adjacent
    # nws = re.compile(r'[^a-zA-Z0-9]')
    # ws = re.compile(r'\s+')
    current = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        # Overlapping or touching - merge them
        if next_span.start <= current.end:
            end = max(current.end, next_span.end)
            current = TextSpan(text=text[current.start : end], start=current.start, end=end)
        elif skip_spaces:
            # Check if there's only whitespace between spans
            between_text = text[current.end : next_span.start]
            # if re.sub(ws, '', between_text.strip()) == "":
            if between_text.isspace() or between_text == "":
                end = max(current.end, next_span.end)
                current = TextSpan(text=text[current.start : end], start=current.start, end=end)
            else:
                # No overlap and more than just whitespace between the spans
                result.append(current)
                current = next_span
        else:
            # No overlap and no need to combine if there are only spaces between the spans
            result.append(current)
            current = next_span
    # Don't forget the last span
    result.append(current)
    # If we combined spans, maybe we need to combine again if there are several of the same
    if size != len(result):
        result = merge_spans(text, result, skip_spaces=skip_spaces)
    result = sorted(result, key=lambda x: x.start)
    return result
