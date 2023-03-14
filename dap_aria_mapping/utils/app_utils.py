"""Utility functions for the front-end of the app
"""
import pandas as pd
import polars as pl

def convert_to_pandas(_df: pl.DataFrame) -> pd.DataFrame:
    """converts polars dataframe to pandas dataframe
    note: this is needed as altair doesn't allow polars, but the conversion is quick so i still think it's 
    worth while to use polars for the filtering

    Args:
        _df (pl.DataFrame): polars dataframe

    Returns:
        pd.DataFrame: pandas dataframe
    """
    return _df.to_pandas()