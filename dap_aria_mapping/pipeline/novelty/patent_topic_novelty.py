""" 
Script to calculate novelty scores at topic level, using patents
Uses typer to create a command line interface

Usage examples:

All levels, full dataset:
python dap_aria_mapping/pipeline/novelty/patent_topic_novelty.py

One level, test dataset:
python dap_aria_mapping/pipeline/novelty/patent_topic_novelty.py --taxonomy-level 1
"""
from dap_aria_mapping import logging, PROJECT_DIR, BUCKET_NAME
from dap_aria_mapping.getters.novelty import (
    get_patent_novelty_scores,
    get_patent_topic_pair_commonness,
)
import dap_aria_mapping.utils.novelty_utils as nu
from dap_aria_mapping.getters.taxonomies import get_topic_names
from nesta_ds_utils.loading_saving.S3 import upload_obj
import typer

OUTPUT_DIR = "outputs/novelty"


def calculate_topic_novelty(
    taxonomy_level: int = 0,
    from_local: bool = False,
    save_to_local: bool = False,
    upload_to_s3: bool = True,
    min_pair_counts: int = 50,
    min_doc_counts: int = 50,
):
    """
    Calculate novelty scores for topics using patent data and uploads them on s3

    Args:
        taxonomy_level (int, optional): Taxonomy level to use for novelty calculation. Must be between 1 and 5. If set to 0, uses all levels
        from_local (bool, optional): Whether to use data from local disk
        save_to_local (bool, optional): Whether to also save the novelty scores to local disk
        min_pair_counts (int, optional): Minimum number of times a topic pair must appear in the corpus to be considered for novelty calculation (for the score based on topic pairs)
        min_doc_counts (int, optional): Minimum number of times a topic must appear in the corpus to be considered for including into the analysis outputs
    """
    if taxonomy_level == 0:
        levels = list(range(1, 6))
    else:
        levels = [taxonomy_level]
    # Loop over taxonomy levels
    for level in levels:
        logging.info(f"Processing taxonomy level {level}")
        # Fetch novelty scores for patents
        patent_novelty_df = get_patent_novelty_scores(
            level=level, from_local=from_local
        ).drop_duplicates("publication_number")
        # Fetch commonness values for topic pairs
        topic_pairs_df = get_patent_topic_pair_commonness(
            level=level, from_local=from_local
        )
        # Fetch topic names
        topic_names = get_topic_names(
            taxonomy_class="cooccur", name_type="chatgpt", level=level
        )
        # Calculate novelty scores for topics
        topic_doc_novelty_df = nu.document_to_topic_novelty(patent_novelty_df, id_column="publication_number")
        topic_pair_novelty_df = nu.pair_to_topic_novelty(
            topic_pairs_df, min_counts=min_pair_counts
        )
        # Merge both topic-level novelty scores
        topic_novelty_df = (
            topic_doc_novelty_df.merge(
                topic_pair_novelty_df, on=["topic", "year"], how="left"
            )
            .assign(topic_name=lambda x: x.topic.map(topic_names))
            .query("doc_counts > @min_doc_counts")
        )
        # Export novelty scores
        filepath_topic_novelty_scores = (
            OUTPUT_DIR + f"/topic_novelty_patents_{level}.parquet"
        )
        if upload_to_s3:
            # Upload to s3
            upload_obj(
                topic_novelty_df,
                BUCKET_NAME,
                f"{filepath_topic_novelty_scores}",
            )
        if save_to_local:
            # Save to local disk
            (PROJECT_DIR / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            export_path = PROJECT_DIR / filepath_topic_novelty_scores
            topic_novelty_df.to_parquet(export_path, index=False)
            logging.info(f"Exported novelty scores to {export_path}")


if __name__ == "__main__":
    typer.run(calculate_topic_novelty)
