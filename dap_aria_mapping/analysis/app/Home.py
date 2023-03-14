import streamlit as st
from streamlit.components.v1 import html
from st_clickable_images import clickable_images
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR

formatting.setup_theme()


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


PAGE_TITLE = "Innovation Explorer"

IMAGE_DIR = f"{PROJECT_DIR}/dap_aria_mapping/analysis/app/images"


# icon to be used as the favicon on the browser tab
nesta_fav = Image.open(f"{IMAGE_DIR}/favicon.ico")

# sets page configuration with favicon and title
st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=nesta_fav)

st.title("Welcome to the Innovation Explorer!")

home_tab, data_tab, methods_tab = st.tabs(["Home", "About the Datasets", "Methodology"])

with home_tab:
    clicked = clickable_images(
        [
            f"https://raw.githubusercontent.com/nestauk/dap_aria_mapping/6a2aef5dfe8d9fe12091b42f8b9d9b422dbe7c8d/dap_aria_mapping/analysis/app/images/hs_homepage.png",
            f"https://raw.githubusercontent.com/nestauk/dap_aria_mapping/6a2aef5dfe8d9fe12091b42f8b9d9b422dbe7c8d/dap_aria_mapping/analysis/app/images/cm_homepage.png",
        ],
        titles=[f"Image #{str(i)}" for i in range(3)],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "400px"},
    )

    if clicked == 0:
        nav_page_from_image("Horizon_Scanner")
    elif clicked == 1:
        nav_page_from_image("Change_Makers")


with data_tab:
    st.markdown(
        "In this app we leverage open source data provided by [Google Patents](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data?pli=1) and [Openalex](https://docs.openalex.org/) to assess the landscape of innovation in the UK"
    )
    st.markdown("ADD MORE DATA DOCUMENTATION")

with methods_tab:
    st.markdown("ADD INFORMATION ABOUT OUR METHODOLOGY")

# adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5, 1, 1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))
