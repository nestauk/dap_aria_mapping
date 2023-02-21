import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
formatting.setup_theme()

PAGE_TITLE = "Horizon Scanner"

IMAGE_DIR = f"{PROJECT_DIR}/dap_aria_mapping/analysis/app/images"


#icon to be used as the favicon on the browser tab
icon = Image.open(f"{IMAGE_DIR}/hs_icon.ico")

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
    st.title(":blue[Horizon Scanner]")

st.markdown(":blue[**_Explore patterns and trends in research domains across the UK_**]")

slider, dropdown = st.columns(2)
with slider:
    top_n = st.slider(label = "Show me most productive:" , min_value = 0, max_value = 50)
with dropdown:
    level = st.selectbox(label = "At the following level of granularity:", options = ["Area", "Discipline", "Topic"])

overview_tab, ed_tab, overlaps_tab = st.tabs(["Overview", "Emergence and Disruption", "Overlaps"])

with overview_tab:
    volume, alignment = st.columns(2)
    with volume:
        st.subheader("Volume of Activity Over Time")
    with alignment:
        st.subheader("Trends in Alignment")

with ed_tab:
    emergence, disruption, novel = st.columns(3)
    with emergence:
         st.subheader("Trends in Emergence")
        
    with disruption:
        st.subheader("Trends in Disruption")
    
    with novel:
        st.subheader("Trends in Novelty")

with overlaps_tab:
    heatmap, link_prediction = st.columns(2)
    with heatmap:
        st.subheader("Heatmap of overlaps")
    with link_prediction:
        st.subheader("Link prediction of future overlaps")
    



#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/nesta_aria_logo.png"))