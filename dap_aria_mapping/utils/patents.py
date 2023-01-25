"""
Utils related to patents data
"""
from typing import List


def format_list_of_strings(list_of_strings: List[str]) -> str:
    """Converts a list of strings to a comma separated string
        where each string element is in quotations.

    Args:
        list_of_strings (List[str]): List of strings to be converted.

    Returns:
        str: Comma separated string.
    """
    return ", ".join([f"'{string}'" for string in list_of_strings])
