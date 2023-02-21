import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
formatting.setup_theme()

PAGE_TITLE = "Change Makers"

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

st.markdown(":red[**_Find the people and institutions with the ability to make waves within research areas_**]")

overview_tab, overlaps_tab = st.tabs(["Overview", "Collaboration"])

with overview_tab:
    dropdown1, dropdown2, dropdown3 = st.columns(3)
    with dropdown1:
        group = st.selectbox(label = "Show me the:" , options = ["Individuals", "Research Groups", "Institutions"])
    with dropdown2:
        indicator = st.selectbox(label = "Who are producing research that is:", options = ["Emergent", "Disruptive", "Novel"])
    with dropdown3:
        topics = st.multiselect(label = "In at least one of the following topics:", options = ["Placeholder Topic1", "Placeholder Topic2"])

    quad_chart, stacked_bars = st.columns(2)

    with quad_chart:
        st.subheader("Placeholder for Quad Chart")
    with stacked_bars:
        st.subheader("Placeholder for Stacked Bars")

with overlaps_tab:
    st.subheader("Placeholder for network")


#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/nesta_aria_logo.png"))