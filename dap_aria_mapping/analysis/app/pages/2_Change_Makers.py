import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
formatting.setup_theme()

PAGE_TITLE = "Horizon Scanner"

IMAGE_DIR = f"{PROJECT_DIR}/dap_aria_mapping/analysis/app/images"


#icon to be used as the favicon on the browser tab
icon = Image.open(f"{IMAGE_DIR}/cm_icon.ico")

# sets page configuration with favicon and title
st.set_page_config(
    page_title=PAGE_TITLE, 
    layout="wide", 
    page_icon=icon
)

header1, header2 = st.columns([1,10])
with header1:
    st.image(icon)
with header2:       
    st.title(":red[Change Makers]")

overview_tab, overlaps_tab = st.tabs(["Overview", "Collaboration"])

#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/nesta_aria_logo.png"))