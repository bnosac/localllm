import importlib_resources
from zipfile import ZipFile
import polars as pl

def data_be_parliament() -> pl.DataFrame:
    """
    Dataset from 2017 with Questions asked by members of the Belgian Federal Parliament and the Answers provided to these questions.
    The dataset was extracted from http://data.dekamer.be and contains questions asked by persons in the Belgium Federal parliament and answers given by the departments of the Federal Belgian Ministers.

    The dataset contains the following information:
      - doc_id: a unique identifier
      - depotdat: the date when the question was registered
      - aut_party / aut_person / aut_language: who asked the question and which political party is he/she a member of + the language of the person who asked the question
      - language: the language of the question
      - question: the question itself 
      - question_theme_main: the main theme of the question
      - question_theme: a comma-separated list of all themes the question is about
      - answer: the answer given by the department of the minister
      - answer_deptpres, answer_department, answer_subdepartment: to which ministerial department has the question been raised to and answered by

    Parameters
    --------

    None

    Returns
    --------

    pl.DataFrame
        A polars DataFrame with example data.
        The dataframe has 14034 rows with 7017 questions which are both available in french and dutch. 
        The dataframe has the following columns: 'doc_id', 'depotdat', 'aut_party', 'aut_person', 'aut_language', 'language', 'question', 'question_theme_main', 'question_theme', 'answer', 'answer_deptpres', 'answer_department', 'answer_subdepartment'

    Reference
    ---------
    
    http://data.dekamer.be, data is provided by http://www.dekamer.be in the public domain (CC0).    

    Examples
    --------

    >>> from localllm.data import data_be_parliament
    >>> x = data_be_parliament()
    """
    path = importlib_resources.files("localllm") / "data" / "be_parliament.zip"        
    path = str(path)
    x = pl.read_csv(ZipFile(path).open("be_parliament.csv", mode='r').read(), encoding = "utf-8", infer_schema=False)
    return x
