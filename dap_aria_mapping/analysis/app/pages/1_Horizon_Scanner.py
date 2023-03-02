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
    st.markdown(f'<h1 style="color:#0000FF;font-size:72px;">{"Horizon Scanner"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#0000FF;font-size:16px;">{"<em>Explore patterns and trends in research domains across the UK<em>"}</h1>', unsafe_allow_html=True)

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

overview_tab, disruption_tab, novelty_tab, overlaps_tab = st.tabs(["Overview", "Disruption", "Novelty","Overlaps"])

with overview_tab:
    volume, alignment = st.columns(2)
    with volume:
        st.subheader("Trends in Emergence")
        st.markdown("This would show trends in growth over time for areas/domains/topics, allowing users to analyse patterns recognizing that certain areas produce more/less content than others")
    with alignment:
        st.subheader("Trends in Alignment")
        st.markdown("This could illustrate if research is becoming more/less aligned with industry in certain areas")

with disruption_tab:
    disruption_trends, disruption_drilldown = st.columns(2)
        
    with disruption_trends:
        st.subheader("Trends in Disruption")
        st.markdown("This could show if certain domains/areas have been recently disrupted or have lacked disruption")
    
    with disruption_drilldown:
        st.subheader("Drill Down in Disruption")
        st.markdown("This would allow a user to select a topic and see the distribution of disruptiveness of papers within that topic")

with novelty_tab:
    st.subheader("Trends in Novelty")
    st.markdown("This could show trends in novelty of research produced by certain domains/areas")

with overlaps_tab:
    heatmap, overlap_drilldown = st.columns(2)
    with heatmap:
        st.subheader("Heatmap of Overlaps")
        st.markdown("This would be used to show which areas have a large amount of research that spans multiple topics (i.e. a lot of research combining ML with Neuroscience)")
    with overlap_drilldown:
        st.subheader("Trends in Overlaps")
        st.markdown("This would allow a user to select interesting overlaps and view the trends in growth over time")
    



#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))