import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
formatting.setup_theme()

PAGE_TITLE = "Innovation Explorer"

IMAGE_DIR = f"{PROJECT_DIR}/dap_aria_mapping/analysis/app/images"


#icon to be used as the favicon on the browser tab
nesta_fav = Image.open(f"{IMAGE_DIR}/favicon.ico")

# sets page configuration with favicon and title
st.set_page_config(
    page_title=PAGE_TITLE, 
    layout="wide", 
    page_icon=nesta_fav
)

st.title("Welcome to the Innovation Explorer!")

home_tab, data_tab, methods_tab = st.tabs(["Home", "About the Datasets", "Methodology"])

with home_tab:
    hs, cm = st.columns(2)
    with hs:
        st.image(Image.open(f"{IMAGE_DIR}/hs_homepage.png"))

    with cm:
        st.image(Image.open(f"{IMAGE_DIR}/cm_homepage.png"))

        

with data_tab:
    st.markdown("In this app we leverage open source data provided by [Google Patents](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data?pli=1) and [Openalex](https://docs.openalex.org/) to assess the landscape of innovation in the UK")
    st.markdown("ADD MORE DATA DOCUMENTATION")

with methods_tab:
    st.markdown("ADD INFORMATION ABOUT OUR METHODOLOGY")

#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/nesta_aria_logo.png"))

