import csv
import zipfile
import importlib_resources
from typing import List
from io import StringIO
#from zipfile import ZipFile

def data_be_parliament() -> List[dict]:
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

    list[dict]
        A list of dicts where each list element is a question asked in parliament.
        The dataset has 14034 rows with 7017 questions which are both available in french and dutch. 
        The dataset has the following columns: 'doc_id', 'depotdat', 'aut_party', 'aut_person', 'aut_language', 'language', 'question', 'question_theme_main', 'question_theme', 'answer', 'answer_deptpres', 'answer_department', 'answer_subdepartment' if you convert it to pandas or polars

    Reference
    ---------
    
    http://data.dekamer.be, data is provided by http://www.dekamer.be in the public domain (CC0).    

    Examples
    --------

    >>> from localllm.data import data_be_parliament
    >>> x = data_be_parliament()
    >>> import pprint
    >>> pprint.pprint(x[0], width=1000)                                       # doctest: +SKIP
    {
      'answer': 'Ik nodig het geachte lid uit zijn vraag te stellen aan mijn collega Kris Peeters, minister van Werk en Economie (Vraag nr. 2000 van 14 februari 2018).',        
      'answer_department': "Minister van Middenstand, Zelfstandigen, KMO's, Landbouw en Maatschappelijke Integratie",
      'answer_deptpres': '14',
      'answer_subdepartment': "Middenstand, Zelfstandigen, KMO's, Landbouw en Maatschappelijke Integratie",
      'aut_language': 'french',
      'aut_party': 'MR',
      'aut_person': 'Flahaux, Jean-Jacques',
      'depotdat': '2017-12-29',
      'doc_id': 'http://data.dekamer.be/v0/qrva/54-B144-14-1021-2017201819553',
      'language': 'dutch',
      'question': 'Percentage vrouwen met een eenoudergezin. \r\n\r\n In Wallonie werden de eenoudergezinnen onlangs gescreend. Daaruit is gebleken dat eenoudergezinnen vandaag 12,20 % van de gezinnen in het zuiden van het land vertegenwoordigen en dat het in acht op de tien gevallen over alleenstaande vrouwen gaat.Hebt u, ter aanvulling van die informatie, cijfers voor het hele land? Zo ja, kunt u dan ook in voorkomend geval de cijfers voor het verleden meedelen, zodat ik mij een beeld kan vormen van de evolutie van de problematiek?',
      'question_theme': 'GEZIN  |  ALLEENSTAANDE  |  VROUW',
      'question_theme_main': ''
    }
    >>> 
    >>> import polars as pl                        # doctest: +SKIP
    >>> be = pl.from_records(x)                    # doctest: +SKIP
    >>> import pandas as                           # doctest: +SKIP
    >>> be = pd.DataFrame.from_records(x)          # doctest: +SKIP
    """
    path = importlib_resources.files("localllm") / "data" / "be_parliament.zip"        
    path = str(path)
    #x = pl.read_csv(ZipFile(path).open("be_parliament.csv", mode='r').read(), encoding = "utf-8", infer_schema=False)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        with zip_ref.open("be_parliament.csv") as csv_file:
            csv_content = csv_file.read().decode('utf-8')
            reader = csv.DictReader(StringIO(csv_content))
            x = [row for row in reader]
    return x
