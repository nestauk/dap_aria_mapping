from nesta_ds_utils.loading_saving.S3 import download_obj
from dap_aria_mapping import BUCKET_NAME

def openalex_coocccurrence_topics(level: int) -> dict:
    """returns openalex documents tagged with topics contained in abstracts, 
    where topics were generated using the cooccurrence taxonomy

    Args:
        level (int): Level of the taxonomy to pull

    Returns:
        dict: dictionary of topics per document. 
        Key: document id, Value: list of topics contained in in the abstract of the document
    """

    return download_obj(
    BUCKET_NAME,
    "outputs/docs_with_topics/cooccurrence_taxonomy/openalex/Level_{}.json".format(level),
    download_as="dict",
    )

def patent_coocccurrence_topics(level: int) -> dict:
    """returns patent documents tagged with topics contained in abstracts, 
    where topics were generated using the cooccurrence taxonomy

    Args:
        level (int): Level of the taxonomy to pull

    Returns:
        dict: dictionary of topics per document. 
        Key: document id, Value: list of topics contained in in the abstract of the document
    """

    return download_obj(
    BUCKET_NAME,
    "outputs/docs_with_topics/cooccurrence_taxonomy/patents/Level_{}.json".format(level),
    download_as="dict",
    )