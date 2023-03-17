import streamlit as st
from PIL import Image
from nesta_ds_utils.viz.altair import formatting
from dap_aria_mapping import PROJECT_DIR, IMAGE_DIR
from dap_aria_mapping.getters.app_tables.change_makers import get_collaboration_network, get_topic_domain_area_lookup
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
from typing import Tuple, List
import polars as pl
import math

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

@st.cache_data
def load_networks(dataset: str) -> nx.Graph:
    """get the collaboration network for either patents or publications

    Args:
        dataset (str): "Academia" for publication network or "Industry" for patent network 

    Returns:
        nx.Graph: collaboration network
    """
    return get_collaboration_network(dataset)

@st.cache_data
def load_filter_data(dataset: str) -> Tuple[pl.DataFrame, List[str]]:
    """gets lookup table of domains to areas and topic names to populate filters, 
    as well as initial list of all domains for starting filter value

    Args:
        dataset: "Industry" or "Academia" specified by filter
    Returns:
        Tuple[pl.DataFrame, List[str]]: lookup table for filtering and unique domains
    """
    if dataset == "Industry":
        data = get_topic_domain_area_lookup("patents")
    elif dataset == "Academia":
        data = get_topic_domain_area_lookup("publications")
    unique_domains = data["domain_name"].unique().to_list()
    return data, unique_domains

@st.cache_data
def filter_by_domain(domain: str, dataset: str) -> Tuple[nx.Graph, List[str]]:
    """generates options for the topic filter, and filters the network to only contain nodes of a given domain

    Args:
        domain (str): domain name from filter
        dataset (str): either "Academia" or "Industry"

    Returns:
        Tuple[nx.Graph, List[str]]: subgraph only containing nodes within the domain, and unique areas to populate filter
    """
    #must load the network in the function, otherwise won't be able to go back to another domain
    network = load_networks(dataset)
    filter_data, unique_domains = load_filter_data(dataset)
    unique_areas = filter_data.filter(pl.col("domain_name") == domain)["area_name"].unique().to_list()
    node_domains = nx.get_node_attributes(network, "domains")
    nodes = [node for node in network.nodes() if domain in node_domains[node]["domain_name"]]
    return network.subgraph(nodes), unique_areas

@st.cache_data
def filter_by_area(area: str, _filter_data: pl.DataFrame, _network: nx.Graph) -> Tuple[nx.Graph, List[str]]:
    """generates options for the topic filter, and filters the network to only contain nodes of a given domain

    Args:
        area (str): area name from filter
        filter_data (pl.DataFrame): lookup table for filtering
        _network (nx.Graph): collaboration network

    Returns:
        Tuple[nx.Graph, List[str]]: subgraph only containing nodes within the area, and unique topics to populate filter
    """
    unique_topics = _filter_data.filter(pl.col("area_name") == domain)["topic_name"].unique().to_list()
    node_areas= nx.get_node_attributes(_network, "areas")
    nodes = [node for node in network.nodes() if area in node_areas[node]["area_name"]]
    return network.subgraph(nodes), unique_topics

@st.cache_data
def filter_by_topic(topic: str, _network: nx.Graph) -> nx.Graph:
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
def nx_to_agraph(_network: nx.Graph, filter_val: str, filter_field: str) -> Tuple[List[Node], List[Edge]]:
    """convert networkx graph to lists of streamlit-agraph nodes and edges.
    Only returns a maximum of 500 nodes/edges

    Args:
        network (nx.Graph): networkx graph
        filter_val (str): domain, area, or topic name (specified by filters)
        filter_field (str): domains, areas, or topics depending on the level selected in the filter

    Returns:
        Tuple[List[Node], List[Edge]]: lists of streamlit-agraph nodes and edges
    """
    node_size = nx.get_node_attributes(_network, "overall_doc_count")
    edge_weight = nx.get_edge_attributes(_network, filter_field)
    #log scale node size
    nodes = [
        Node(
            id=node,
            label = node,
            size = math.log(node_size[node]["publication_number"]),
            shape = "dot",
            color = "0000FF")
            for node in _network.nodes()]
    edges = [
        Edge(
            source=edge[0],
            weight=edge_weight[edge][filter_val],
            target=edge[1],
            color = "EB003B"
            ) for edge in _network.edges()]
    if len(nodes)>500:
        print("Warning: too many nodes to display, only displaying 500 nodes and edges")
        return nodes[:500], edges[:500]
    else:
        return nodes, edges


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
        group = st.selectbox(label = "Show me the:" , options = ["Individuals", "Research Groups", "Institutions"])
    with dropdown2:
        indicator = st.selectbox(label = "Who are producing research that is:", options = ["Emergent", "Disruptive", "Novel"])
    with dropdown3:
        topics = st.multiselect(label = "In at least one of the following topics:", options = ["Placeholder Topic1", "Placeholder Topic2"])

    quad_chart, stacked_bars = st.columns(2)

    with quad_chart:
        st.subheader("{} {}".format(indicator, group))
        st.markdown("This would be used to show where individuals/institutions are on the axis of high-low disruptiveness (etc.) vs. volume within certain topics. PDs may want to find {} that haven't put out much research yet (low volume) but that research is highly {}.".format(group.lower(),indicator.lower()))
    with stacked_bars:
        st.subheader("Overlaps")
        st.markdown("This would show a breakdown of other research topics that the {} are looking at to help illustrate where there are opportunities for collaboration".format(group.lower()))

with collaboration_tab:
    network_dropdown1, network_dropdown2 = st.columns(2)
    with network_dropdown1:
        dataset = st.selectbox(label = "Show me relationships in", options = ["Industry", "Academia"])
    
    #with network_dropdown2:
    #    if dataset == "Academia":
    #        groups = st.selectbox(label = "Explore relationships between", options = ["Individuals", "Research Groups", "Institutions"])
    #    elif dataset == "Industry":
    #        groups = st.selectbox(label = "Explore relationships between", options = ["Individuals", "Companies"])

    filter_data, unique_domains = load_filter_data(dataset)
    with st.sidebar:
        level = "domain"
        domain = st.selectbox(label = "Select a Domain", options = unique_domains)
        #NOTE: YOU MUST SELECT A DOMAIN

        #allow user to filter by area, but also allow "all"
        network, unique_areas = filter_by_domain(domain, dataset)
        unique_areas.insert(0, "All")
        nodes, edges = nx_to_agraph(network, domain, filter_field = "domains")
        area = st.selectbox(label = "Select an Area", options = unique_areas)
        if area != "All":
            level = "area"
            #if an area is selected, filter data to the area and allow user to filter by topic or select "all"
            network, unique_topics = filter_by_area(area, filter_data, network)
            nodes, edges = nx_to_agraph(network, area, filter_field = "areas")
            unique_topics.insert(0, "All")
            topic = st.selectbox(label = "Select a Topic", options = unique_topics)

            if topic != "All":
                level = "topic"
                network = filter_by_topic(topic, network)
                nodes, edges = nx_to_agraph(network, topic, filter_field = "topics")

    st.subheader("Collaboration Network")
    
    config = Config(width=1100,
                height=950,
                directed=False, 
                physics=True, 
                hierarchical=False,
                # **kwargs
                )

    agraph(nodes, edges, config)


#adds the nesta x aria logo at the bottom of each tab, 3 lines below the contents
st.markdown("")
st.markdown("")
st.markdown("")

white_space, logo, white_space = st.columns([1.5,1,1.5])
with logo:
    st.image(Image.open(f"{IMAGE_DIR}/igl_nesta_aria_logo.png"))
