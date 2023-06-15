import streamlit as st
from st_click_detector import click_detector
from streamlit.components.v1 import html
from PIL import Image
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
from dap_aria_mapping.utils.app_utils import (
    img_to_bytes,
    nav_page_from_image,
    create_hover_class,
)

formatting.setup_theme()

PAGE_TITLE = "DEMO - Innovation Explorer"

IMAGE_DIR = f"{PROJECT_DIR}/dap_aria_mapping/analysis/app/images"

# icon to be used as the favicon on the browser tab
nesta_fav = Image.open(f"{IMAGE_DIR}/favicon.ico")

# sets page configuration with favicon and title
st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=nesta_fav)

st.title("Welcome to the DEMO Innovation Explorer!")

home_tab, data_tab, methods_tab = st.tabs(["Home", "About the Datasets", "Methodology"])

with home_tab:

    hs_img, cm_img = (
        img_to_bytes(f"{IMAGE_DIR}/hs_homepage.png"),
        img_to_bytes(f"{IMAGE_DIR}/cm_homepage.png"),
    )

    classes_images = {
        "img-acu-1": {
            "png": "https://s2.gifyu.com/images/hs_homepage.png",
            "gif": "https://s2.gifyu.com/images/hs_homepage.gif",
        },
        "img-acu-2": {
            "png": "https://s10.gifyu.com/images/cm_homepage.png",
            "gif": "https://s10.gifyu.com/images/cm_homepage.gif",
        },
    }

    for key, value in classes_images.items():
        create_hover_class(key, value["png"], value["gif"])

    content = """
        <div style="display: flex; justify-content: center; margin: 0 auto; padding: 10px 0;">
        <a href='#' id='img-1'><img width='90%' class='{acu1}' src='data:image/png;base64,{hs_img}'></a>
        <a href='#' id='img-2'><img width='90%' class='{acu2}' src='data:image/png;base64,{cm_img}'></a>
        </div>
    """.format(
        hs_img=hs_img, cm_img=cm_img, acu1="img-acu-1", acu2="img-acu-2"
    )

    clicked = click_detector(content)

    if clicked == "img-1":
        nav_page_from_image(page="Horizon_Scanner", timeout=5)
    elif clicked == "img-2":
        nav_page_from_image(page="Change_Makers", timeout=5)


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
# with logo:
#     st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))
