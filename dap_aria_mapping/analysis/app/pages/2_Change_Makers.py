import streamlit as st
from PIL import Image
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR, IMAGE_DIR
from dap_aria_mapping.getters.app_tables.change_makers import get_collaboration_network, get_quad_chart_data
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
from typing import Tuple, List
import polars as pl
import math
import altair as alt

formatting.setup_theme()

PAGE_TITLE = "Change Makers"

#icon to be used as the favicon on the browser tab
icon = Image.open(f"{IMAGE_DIR}/cm_icon.ico")

# sets page configuration with favicon and title
st.set_page_config(
    page_title=PAGE_TITLE, 
    layout="wide", 
    page_icon=icon
)

@st.cache_data(show_spinner = "Loading Data")
def load_networks(dataset: str) -> nx.Graph:
    """get the collaboration network for either patents or publications

    Args:
        dataset (str): "Academia" for publication network or "Industry" for patent network 

    Returns:
        nx.Graph: collaboration network
    """
    return get_collaboration_network(dataset)

@st.cache_data(show_spinner = "Filtering Data to Domain")
def filter_network_by_domain(domain: str, dataset: str) -> Tuple[nx.Graph, List[str]]:
    """generates options for the topic filter, and filters the network to only contain nodes of a given domain

    Args:
        domain (str): domain name from filter
        dataset (str): either "Academia" or "Industry"

    Returns:
        Tuple[nx.Graph, List[str]]: subgraph only containing nodes within the domain, and unique areas to populate filter
    """
    #must load the network in the function, otherwise won't be able to go back to another domain
    network = load_networks(dataset)
    node_domains = nx.get_node_attributes(network, "domains")
    nodes = [node for node in network.nodes() if domain in node_domains[node]["domain_name"]]
    return network.subgraph(nodes) 

@st.cache_data(show_spinner = "Filtering Data to Area")
def filter_network_by_area(area: str, _network: nx.Graph) -> Tuple[nx.Graph, List[str]]:
    """generates options for the topic filter, and filters the network to only contain nodes of a given domain

    Args:
        area (str): area name from filter
        filter_data (pl.DataFrame): lookup table for filtering
        _network (nx.Graph): collaboration network

    Returns:
        Tuple[nx.Graph, List[str]]: subgraph only containing nodes within the area, and unique topics to populate filter
    """
    node_areas= nx.get_node_attributes(_network, "areas")
    nodes = [node for node in network.nodes() if area in node_areas[node]["area_name"]]
    return network.subgraph(nodes)

@st.cache_data
def filter_network_by_topic(topic: str, _network: nx.Graph) -> nx.Graph:
    """generates options for the topic filter, and filters the network to only contain nodes of a given domain

    Args:
        topic (str): topic name from filter
        _network (nx.Graph): collaboration network

    Returns:
       nx.Graph : subgraph only containing nodes within the topic
    """
    node_topics= nx.get_node_attributes(_network, "topics")
    nodes = [node for node in network.nodes() if topic in node_topics[node]["topic_name"]]
    return network.subgraph(nodes)

@st.cache_data
def nx_to_agraph(_network: nx.Graph, filter_val: str, filter_field: str, id_col: str) -> Tuple[List[Node], List[Edge]]:
    """convert networkx graph to lists of streamlit-agraph nodes and edges.

    Args:
        network (nx.Graph): networkx graph
        filter_val (str): domain, area, or topic name (specified by filters)
        filter_field (str): domains, areas, or topics depending on the level selected in the filter
        id_col (str): either "id" for publications for "publication_number" for patents

    Returns:
        Tuple[List[Node], List[Edge]]: lists of streamlit-agraph nodes and edges
    """
    node_size = nx.get_node_attributes(_network, "overall_doc_count")
    edge_weight = nx.get_edge_attributes(_network, filter_field)

    nodes = [
        Node(
            id=node,
            label = node,
            size = math.log(node_size[node][id_col]),
            shape = "dot",
            color = "#0f294a")
            for node in _network.nodes()]
    edges = [
        Edge(
            source=edge[0],
            weight=edge_weight[edge][filter_val],
            target=edge[1],
            color = "EB003B"
            ) for edge in _network.edges()]
            
    return nodes, edges

@st.cache_data(show_spinner = "Loading quad chart")
def load_quad_data():
    """gets the disruption and volume by institution to populate the quad chart

    Returns:
        pl.DataFrame: polars dataframe with columns 
        ["domain_name", "area_name", "topic_name", "affiliation_string", "volume", "average_cd_score", "average_novelty_score"]
    """
    quad_data = get_quad_chart_data()
    domain_filter_options = quad_data["domain_name"].unique().to_list()
    return quad_data, domain_filter_options

@st.cache_data(show_spinner = "Filtering by domain")
def filter_quad_by_domain(_quad_data: pl.DataFrame, domain: str) -> Tuple[pl.DataFrame, List[str]]:
    """filters the quad chart data by a domain

    Args:
        _quad_data (pl.DataFrame): quad chart data
        domain (str): domain selected by filter

    Returns:
        Tuple[pl.DataFrame, List[str]]: filtered quad chart data, list of unique areas to populate filter
    """
    quad_data = _quad_data.filter(
        pl.col("domain_name")==domain
        )

    unique_areas = quad_data["area_name"].unique().to_list()
    return quad_data, unique_areas

@st.cache_data(show_spinner = "Filtering by area")
def filter_quad_by_area(_quad_data: pl.DataFrame, area: str) -> Tuple[pl.DataFrame, List[str]]:
    """filters the quad chart data by a topic

    Args:
        _quad_data (pl.DataFrame): quad chart data
        area (str): area selected by filter

    Returns:
        Tuple[pl.DataFrame, List[str]]: filtered quad chart data, list of unique topics to populate filter
    """
    quad_data = _quad_data.filter(
        pl.col("area_name")==area
        )
    unique_topics = quad_data["topic_name"].unique().to_list()

    return quad_data, unique_topics

@st.cache_data(show_spinner = "Filtering by topic")
def filter_quad_by_topic(_quad_data: pl.DataFrame, topic: str) -> pl.DataFrame:
    """filters the quad data by an area

    Args:
        _quad_data (pl.DataFrame): quad chart data
        topic (str): topic selected by filter

    Returns:
        pl.DataFrame: filtered quad chart data
    """
    return _quad_data.filter(pl.col("topic_name")==topic)


def group_quad_data(_quad_data: pl.DataFrame, select_field: str) -> pl.DataFrame:
    """groups the quad chart data by affiliaton string and calculates the total volume and 
        average disruptiveness or novelty

    Args:
        _quad_data (pl.DataFrame): quad chart data
        select_field (str): field to visualize (either c-d score or novelty score) specified by filter

    Returns:
        pl.DataFrame: grouped quad chart data
    """
    q = (
    _quad_data.lazy()
    .select(
            [pl.col("affiliation_string"),
            pl.col("volume"),
            pl.col(select_field)
            ]
        )
    .drop_nulls()
    .groupby(["affiliation_string"])
    .agg([
        pl.sum("volume").alias("volume"),
        pl.mean(select_field).alias(select_field)
        ])
    )
    return q.collect()
    
header1, header2 = st.columns([1,10])
with header1:
    st.image(icon)
with header2:       
    st.markdown(f'<h1 style="color:#EB003B;font-size:72px;">{"Change Makers"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#EB003B;font-size:16px;">{"<em>Find the people and institutions with the ability to make waves within research areas<em>"}</h1>', unsafe_allow_html=True)

overview_tab, collaboration_tab = st.tabs(["Overview", "Collaboration"])

with overview_tab:
    dropdown1, dropdown2, dropdown3 = st.columns(3)
    with dropdown1:
        #In theory this would say "Individuals", "Institutions", or "Research Groups"
        group = st.selectbox(label = "Show me the:" , options = ["Institutions"])
    with dropdown2:
        #Currently only showing disruptive
        indicator = st.selectbox(label = "Who are producing research that is:", options = ["Disruptive", "Novel"])
    
    if indicator == "Disruptive":
        select_field = "average_cd_score"
        axis_title = "Average C-D Score"

    elif indicator == "Novel":
        select_field = "average_novelty_score"
        axis_title = "Average Novelty Score"

    
    quad_data, domain_filter_options = load_quad_data()

    with st.sidebar:
        domain = st.selectbox(label = "Select a Domain", options = domain_filter_options)
        area = "All"
        topic = "All"
        quad_data, area_filter_options = filter_quad_by_domain(quad_data, domain)
        area_filter_options.insert(0, "All")
        area = st.selectbox(label = "Select a Area", options = area_filter_options)

        if area != "All":
            quad_data, topic_filter_options = filter_quad_by_area(quad_data, area)
            topic_filter_options.insert(0, "All")
            topic = st.selectbox(label = "Select a Topic", options = topic_filter_options)
            
            if topic != "All":

                quad_data = filter_quad_by_topic(quad_data, topic)
    
    quad_data_to_plot = group_quad_data(quad_data, select_field).to_pandas()
    quad_chart = alt.Chart(quad_data_to_plot).mark_circle().encode(
        x = alt.X("volume:Q", axis = alt.Axis(tickCount = 2, title = "Total Publications"), scale = alt.Scale(domain = [0, quad_data_to_plot["volume"].max()])),
        y = alt.Y("{}:Q".format(select_field), axis = alt.Axis(tickCount = 2, title = axis_title)),
        tooltip = [
            alt.Tooltip(field = "affiliation_string", title = "Organisation"),
            alt.Tooltip(field = "volume", title = "Total Publications"),
            alt.Tooltip(field = select_field, title = axis_title)]
    ).properties(width = 1200, height = 500).configure_mark(color = "#0000FF").interactive()

    st.altair_chart(quad_chart)


with collaboration_tab:
    network_dropdown1, network_dropdown2 = st.columns(2)
    with network_dropdown1:
        dataset = st.selectbox(label = "Show me relationships in", options = ["Industry", "Academia"])
        if dataset == "Academia":
            id_col = "id"
            org_name_col = "affiliation_string"
        else:
            id_col = "publication_number"
            org_name_col = "assignee_harmonized_names"
    
    level = "domain"
    #NOTE: YOU MUST SELECT A DOMAIN

    #allow user to filter by area, but also allow "all"
    no_network = False
    network = filter_network_by_domain(domain, dataset)
    nodes, edges = nx_to_agraph(network, domain, filter_field = "domains", id_col = id_col)
    if area != "All":
        level = "area"
        #if an area is selected, filter data to the area and allow user to filter by topic or select "all"
        network = filter_network_by_area(area, network)
        try:
            nodes, edges = nx_to_agraph(network, area, filter_field = "areas", id_col= id_col)
        except KeyError:
            no_network = True
            st.markdown("No groups in the top 500 institutions in {} produced research in {}".format(dataset, area))

        if topic != "All":
            level = "topic"
            network = filter_network_by_topic(topic, network)
            try:
                nodes, edges = nx_to_agraph(network, topic, filter_field = "topics", id_col=id_col)
            except KeyError:
                no_network = True
                st.markdown("No groups in the top 500 institutions in {} produced research in {}".format(dataset, area))

    st.subheader("Collaboration Network")
    st.markdown("🚨 This view is currently **experimental** and only shows relationships in the most productive 500 institutions in patents and publications")
    
    config = Config(width=1100,
                height=950,
                directed=False, 
                physics=True, 
                hierarchical=False,
                # **kwargs
                )

    if not no_network:
        agraph(nodes, edges, config)


#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))
