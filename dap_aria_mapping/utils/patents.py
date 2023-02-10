"""
Utils related to collecting patents data
"""
from typing import List


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
