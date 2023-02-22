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

area_drop, discipline_drop, topic_drop = st.columns(3)
    
with area_drop:
    area = st.selectbox(label = "Select an Area", options = ["All", "Area 1", "Area 2"])
    discipline = "All"
    topic = "All"

with discipline_drop:
    if area != "All":
        #In reality, the options for discipline would come from df.loc[df["Level 1"] == area]["Level 2"].unique()
        discipline = st.selectbox(label = "Select a Discipline", options = ["All", "Discipline 1", "Discipline 2"])

with topic_drop:
    if discipline != "All":
        #In reality, the options for discipline would come from df.loc[df["Level 2"] == discipline]["Level 3"].unique()
        topic = st.selectbox(label = "Select a Topic", options = ["All", "Topic 1", "Topic 2"])


total_to_display = st.slider(label = "Show me most productive:" , min_value = 0, max_value = 50)

overview_tab, nd_tab, emergence_tab, overlaps_tab = st.tabs(["Overview", "Novelty and Disruption", "Emergence","Overlaps"])

with overview_tab:
    volume, alignment = st.columns(2)
    with volume:
        st.subheader("Volume of Activity Over Time")
    with alignment:
        st.subheader("Trends in Alignment")

with nd_tab:
    disruption, novelty = st.columns(2)
        
    with disruption:
        st.subheader("Trends in Disruption")
    
    with novelty:
        st.subheader("Trends in Novelty")

with emergence_tab:
    st.subheader("Placeholder for exploring trends in emergence")
    st.markdown("Note: this is not clearly defined what this would actually show")

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