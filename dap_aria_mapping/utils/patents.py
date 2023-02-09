"""
Utils related to collecting patents data
"""
from typing import List

# from google patents data dictionary
CITATION_MAPPER = {
    "A": "technological background",
    "D": "document cited in application",
    "E": "earlier patent document",
    "1": "document cited for other reasons",
    "O": "Non-written disclosure",
    "P": "Intermediate document",
    "T": "theory or principle",
    "X": "relevant if taken alone",
    "Y": "relevant if combined with other documents",
    "CH2": "Chapter 2",
    "SUP": "Supplementary search report",
    "ISR": "International search report",
    "SEA": "Search report",
    "APP": "Applicant",
    "EXA": "Examiner",
    "OPP": "Opposition",
    "115": "article 115",
    "PRS": "Pre-grant pre-search",
    "APL": "Appealed",
    "FOP": "Filed opposition",
}

# from google docs about size of query
MAX_SIZE = 1024000

# thresholding focal ids based on publication year dates
EARLY_YEAR, LATE_YEAR = 2007, 2017


def chunk(lst: List[str], n: int) -> List[List[str]]:
    """Chunk a list of strings into n lists of strings.

    Args:
        lst (List[str]): List of strings to be chunked.
        n (int): Number of elements per chunk.

    Returns:
        List[List[str]]: List of lists of chunked strings.
    """
    return [lst[i::n] for i in range(n)]


def format_list_of_strings(list_of_strings: List[str]) -> str:
    """Converts a list of strings to a comma separated string
        where each string element is in quotations.

    Args:
        list_of_strings (List[str]): List of strings to be converted.

    Returns:
        str: Comma separated string.
    """
    return ", ".join([f"'{string}'" for string in list_of_strings])
