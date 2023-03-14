"""Utility functions for the front-end of the app
"""
import pandas as pd
import polars as pl
import base64
from pathlib import Path
from streamlit.components.v1 import html

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

def nav_page_from_image(page: str, timeout: int = 5) -> None:
    """Navigates to a page in the Streamlit app.

    Args:
        page (str): The name of the page to navigate to.
        timeout (int, optional): The number of seconds to wait before timing out
            the navigation. Defaults to 5.
    """
    nav_script = """
            <script type="text/javascript">
                function nav_page(page, start_time, timeout) {
                    var links = window.parent.document.getElementsByTagName("a");
                    for (var i = 0; i < links.length; i++) {
                        if (links[i].href.toLowerCase().endsWith("/" + page.toLowerCase())) {
                            links[i].click();
                            return;
                        }
                    }
                    var elasped = new Date() - start_time;
                    if (elasped < timeout * 1000) {
                        setTimeout(nav_page, 100, page, start_time, timeout);
                    } else {
                        alert("Unable to navigate to page '" + page + "' after " + timeout + " second(s).");
                    }
                }
                window.addEventListener("load", function() {
                    nav_page("%s", new Date(), %d);
                });
            </script>
        """ % (
        page,
        timeout,
    )
    html(nav_script)

def img_to_bytes(img_path: str) -> str:
    """Converts an image to a base64 encoded string.

    Args:
        img_path (str): The path to the image.

    Returns:
        str: The base64 encoded string.
    """
    img_bytes = Path(img_path).read_bytes()
    encoded_img = base64.b64encode(img_bytes).decode()
    return encoded_img