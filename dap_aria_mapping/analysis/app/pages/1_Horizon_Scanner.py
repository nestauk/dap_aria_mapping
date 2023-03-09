import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
from dap_aria_mapping.getters.app_tables.horizon_scanner import volume_per_year
import polars as pl
import pandas as pd
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

#must wrap getter in another function so streamlit doesn't load the file in each time
@st.cache_data
def load_volume_data():
    volume_data = volume_per_year()
    volume_data = volume_data.with_columns(
        (pl.col('patent_count') + pl.col('publication_count')).alias('total_docs'),
        pl.col('year').round(0)
    )
    unique_domains = list(list(volume_data.select(pl.col("domain").unique()))[0])
    unique_domains.insert(0,"All")
    return volume_data, unique_domains


@st.cache_data
def filter_by_domain(domain, _volume_data):
    volume_data = _volume_data.filter(pl.col("domain")==domain)
    unique_areas = list(list(volume_data.select(pl.col("area").unique()))[0])
    return volume_data, unique_areas

@st.cache_data
def filter_by_area(area, _volume_data):
    volume_data = _volume_data.filter(pl.col("area")==area)
    unique_topics = list(list(volume_data.select(pl.col("topic").unique()))[0])
    return volume_data, unique_topics

#@st.cache_data
def group_by_level(_volume_data, level, sum_col):
    q = (_volume_data.lazy().with_columns(pl.col(level).cast(str)).groupby(["year", level]).agg([pl.sum(sum_col)]).filter(pl.any(pl.col("year").is_not_null())))
    return q.collect()

#@st.cache_data
def convert_to_pandas(_df: pl.DataFrame) -> pd.DataFrame:
    return _df.to_pandas()

header1, header2 = st.columns([1,10])
with header1:
    st.image(icon)
with header2:       
    st.markdown(f'<h1 style="color:#0000FF;font-size:72px;">{"Horizon Scanner"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#0000FF;font-size:16px;">{"<em>Explore patterns and trends in research domains across the UK<em>"}</h1>', unsafe_allow_html=True)

#load in volume data 
volume_data, unique_domains = load_volume_data()

#unique_domains.insert(0, "All")

domain_drop, area_drop = st.columns(2)

with domain_drop:
    domain = st.selectbox(label = "Select a Domain", options = unique_domains)
    area = "All"
    topic = "All"
    level_considered = "domain"
    if domain != "All":
        volume_data, unique_areas = filter_by_domain(domain, volume_data)
        level_considered = "area"

    
with area_drop:
    if domain != "All":
        unique_areas.insert(0, "All")
        area = st.selectbox(label = "Select an Area", options = unique_areas)
        if area != "All":
            volume_data, unique_topics = filter_by_area(area, volume_data)
            level_considered = "topic"


#total_to_display = st.slider(label = "Show me most productive:" , min_value = 0, max_value = 50)

overview_tab, disruption_tab, novelty_tab, overlaps_tab = st.tabs(["Overview", "Disruption", "Novelty","Overlaps"])

with overview_tab:
    volume, alignment = st.columns(2)
    with volume:
        st.subheader("Emergence")
        show_only = st.selectbox(label = "Show Emergence In:", options = ["Publications", "Patents", "Both"])
        if show_only == "Publications":
            y_col = "publication_count"
        elif show_only == "Patents":
            y_col = "patent_count"
        else:
            y_col = "total_docs"

        emergence_data = convert_to_pandas(group_by_level(volume_data, level_considered, y_col))

        volume_chart = alt.Chart(emergence_data).mark_line().encode(
            alt.X("year:N"),
            alt.Y("{}:Q".format(y_col)),
            color = "{}:N".format(level_considered)
        ).interactive().properties(height=600)
    st.altair_chart(volume_chart)

    with alignment:
        st.subheader("Alignment")
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