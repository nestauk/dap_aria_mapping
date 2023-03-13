import streamlit as st
from PIL import Image
import altair as alt
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR
from dap_aria_mapping.getters.app_tables.horizon_scanner import volume_per_year
from dap_aria_mapping.getters.taxonomies import get_topic_names
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

@st.cache_data
def load_overview_data():
    volume_data = volume_per_year()

    #add total document count as count of patents and publications combined
    volume_data = volume_data.with_columns(
        (pl.col('patent_count') + pl.col('publication_count')).alias('total_docs'),
        pl.col('year').round(0)
    )

    #add chatgpt names for domain, area, topics
    domain_names  = pl.DataFrame(pd.DataFrame.from_dict(get_topic_names("cooccur", "chatgpt", 1, n_top = 35), orient= "index").rename_axis("domain").reset_index().rename(columns = {"name": "domain_name"})[["domain", "domain_name"]])
    area_names  = pl.DataFrame(pd.DataFrame.from_dict(get_topic_names("cooccur", "chatgpt", 2, n_top = 35), orient= "index").rename_axis("area").reset_index().rename(columns = {"name": "area_name"})[["area", "area_name"]])
    topic_names  = pl.DataFrame(pd.DataFrame.from_dict(get_topic_names("cooccur", "chatgpt", 3, n_top = 35), orient= "index").rename_axis("topic").reset_index().rename(columns = {"name": "topic_name"})[["topic", "topic_name"]])

    volume_data = volume_data.join(domain_names, on="domain", how="left")
    volume_data = volume_data.join(area_names, on="area", how="left")
    volume_data = volume_data.join(topic_names, on="topic", how="left")

    unique_domains = list(list(volume_data.select(pl.col("domain_name").unique()))[0])
    unique_domains.insert(0,"All")

    #reformat the patent/publication counts to long form for the alignment chart
    alignment_data = volume_data.melt(id_vars = ["year", "topic", "topic_name","area", "area_name","domain", "domain_name"], value_vars = ["publication_count", "patent_count"])
    alignment_data.columns = ["year", "topic", "topic_name", "area", "area_name", "domain", "domain_name","doc_type", "count"]

    return volume_data, alignment_data, unique_domains

@st.cache_data
def filter_by_domain(domain, _volume_data, _alignment_data):
    volume_data = _volume_data.filter(pl.col("domain_name")==domain)
    alignment_data = _alignment_data.filter(pl.col("domain_name")==domain)
    unique_areas = list(list(volume_data.select(pl.col("area_name").unique()))[0])
    return volume_data, alignment_data, unique_areas

@st.cache_data
def filter_by_area(area, _volume_data, _alignment_data):
    volume_data = _volume_data.filter(pl.col("area_name")==area)
    alignment_data = _alignment_data.filter(pl.col("area_name")==area)
    unique_topics = list(list(volume_data.select(pl.col("topic_name").unique()))[0])
    return volume_data, alignment_data, unique_topics

def group_emergence_by_level(_volume_data, level, y_col):
    q = (_volume_data.lazy().with_columns(
        pl.col(level).cast(str)
        ).groupby(
            [level, "{}_name".format(level),"year"]
            ).agg(
                [pl.sum(y_col)]
                ).filter(pl.any(pl.col("year").is_not_null())))
    return q.collect()

def group_alignment_by_level(_alignment_data, level):
    total_pubs = _alignment_data.filter(pl.col("doc_type")=="publication_count").select(pl.sum("count"))
    total_patents = _alignment_data.filter(pl.col("doc_type")=="patent_count").select(pl.sum("count"))
    q = (_alignment_data.lazy().with_columns(
        pl.col(level).cast(str)
        ).groupby(["doc_type", level, "{}_name".format(level)]
        ).agg(
            [pl.sum("count").alias("total")]
        ).with_columns(
            pl.when(pl.col("doc_type") == "publication_count")
            .then(pl.col("total")/total_pubs)
            .when(pl.col("doc_type") == "patent_count")
            .then(pl.col("total")/total_patents)
            .alias("doc_fraction")
        ))
    return q.collect()

def convert_to_pandas(_df: pl.DataFrame) -> pd.DataFrame:
    return _df.to_pandas()

header1, header2 = st.columns([1,10])
with header1:
    st.image(icon)
with header2:       
    st.markdown(f'<h1 style="color:#0000FF;font-size:72px;">{"Horizon Scanner"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#0000FF;font-size:16px;">{"<em>Explore patterns and trends in research domains across the UK<em>"}</h1>', unsafe_allow_html=True)

#load in volume data 
volume_data, alignment_data, unique_domains = load_overview_data()

#domain_drop, area_drop = st.columns(2)

with st.sidebar:
#with domain_drop:
    domain = st.selectbox(label = "Select a Domain", options = unique_domains)
    area = "All"
    topic = "All"
    level_considered = "domain"
    if domain != "All":
        volume_data, alignment_data, unique_areas = filter_by_domain(domain, volume_data, alignment_data)
        level_considered = "area"

    
#with area_drop:
    if domain != "All":
        unique_areas.insert(0, "All")
        area = st.selectbox(label = "Select an Area", options = unique_areas)
        if area != "All":
            volume_data, alignment_data, unique_topics  = filter_by_area(area, volume_data, alignment_data)
            level_considered = "topic"


#total_to_display = st.slider(label = "Show me most productive:" , min_value = 0, max_value = 50)

overview_tab, disruption_tab, novelty_tab, overlaps_tab = st.tabs(["Overview", "Disruption", "Novelty","Overlaps"])

with overview_tab:

    st.subheader("Growth Over Time")
    st.markdown("View trends in volume of content over time to detect emerging or stagnant areas of innovation")
    show_only = st.selectbox(label = "Show Emergence In:", options = ["All Documents", "Publications", "Patents"])
    if show_only == "Publications":
        y_col = "publication_count"
    elif show_only == "Patents":
        y_col = "patent_count"
    else:
        y_col = "total_docs"

    emergence_data = convert_to_pandas(group_emergence_by_level(volume_data, level_considered, y_col))
    print(emergence_data.head())

    volume_chart = alt.Chart(emergence_data).mark_line().encode(
        alt.X("year:N"),
        alt.Y("{}:Q".format(y_col), title = "Total Documents Published"),
        color = "{}_name:N".format(level_considered)
    ).interactive().properties(width=1000, height = 500)
    st.altair_chart(volume_chart)

    st.subheader("Alignment in Research and Industry")
    st.markdown("Areas with high publication count and low patent count indicates there is significantly more activity in academia than industry on this topic (or vice versa).")
    filtered_alignment_data = convert_to_pandas(group_alignment_by_level(alignment_data, level_considered))
    alignment_chart = alt.Chart(filtered_alignment_data).transform_filter(
        alt.datum.doc_fraction > 0  
        ).mark_point().encode(
        #alt.X("{}:N".format(level_considered), title=None, axis=None),
        alt.Y("doc_fraction:Q", 
            title = "Fraction of Documents of the Given Type", 
            scale=alt.Scale(type="log"), 
            axis = alt.Axis(tickSize=0)),
        column = alt.Column("{}_name:N".format(level_considered), 
            title=None, 
            header = alt.Header(labelFontSize=10, labelOrient = 'bottom', labelAngle = -45, labelAnchor = "end")),
        color = alt.Color("doc_type:N", legend=alt.Legend(
            direction='horizontal',
            legendX=10,
            legendY=-80,
            orient = 'none',
            titleAnchor='middle',
            title = None)
        )).interactive().properties(width = 20)
    st.altair_chart(alignment_chart)

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