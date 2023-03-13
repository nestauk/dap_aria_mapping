""" 
Script to calculate novelty scores for OpenAlex papers
Uses typer to create a command line interface

Usage examples:

All levels, full dataset:
python dap_aria_mapping/pipeline/novelty/calculate_openalex_novelty.py

One level, test dataset:
python dap_aria_mapping/pipeline/novelty/calculate_openalex_novelty.py --taxonomy-level 1 --test
"""
from dap_aria_mapping import logging, PROJECT_DIR, BUCKET_NAME
import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.utils.novelty_utils as nu
from nesta_ds_utils.loading_saving.S3 import upload_obj
import typer

OUTPUT_DIR = "outputs/novelty"


def calculate_openalex_novelty(
    taxonomy_level: int = 0,
    test: bool = False,
    n_test_sample: int = 1000,
    save_to_local: bool = False,
):
    """
    Calculate novelty scores for OpenAlex papers and uploads them on s3

    Args:
        taxonomy_level (int): Taxonomy level to use for novelty calculation. Must be between 1 and 5. If set to 0, uses all levels
        test (bool): Whether to run in test mode (using a small sample of papers)
        n_test_sample (int): Number of papers to use for testing
        save_to_local (bool): Whether to also save the novelty scores to local disk

    Returns:
        None (saved novelty scores to file)
    """
    # Fetch OpenAlex metadata
    works_df = oa.get_openalex_works()
    logging.info(f"Downloaded {len(works_df)} works")
    # Decide on the taxonomy levels to use
    if taxonomy_level == 0:
        levels = list(range(1, 6))
    else:
        levels = [taxonomy_level]
    # Loop over taxonomy levels
    for level in levels:
        logging.info(f"Processing taxonomy level {level}")
        # Load topics for all papers
        topics_dict = oa.get_openalex_topics(level=level)
        topics_df = nu.preprocess_topics_dict(
            topics_dict, works_df, "work_id", "publication_year"
        )
        logging.info(
            f"Using {len(topics_df)} papers with sufficient data for novelty calculation"
        )
        # Subsample for testing
        test_suffix = ""
        if test:
            topics_df = topics_df.sample(n_test_sample)
            test_suffix = f"_test_n{n_test_sample}"
            logging.info(f"TEST MODE: Using {len(topics_df)} papers for testing")
        # Calculate novelty scores
        work_novelty_df, topic_pair_commonness_df = nu.document_novelty(
            topics_df, "work_id"
        )
        # Export novelty scores
        filepath_document_novelty_scores = (
            OUTPUT_DIR + f"/openalex_novelty_{level}{test_suffix}.parquet"
        )
        filepath_topic_pair_commonness = (
            OUTPUT_DIR + f"/openalex_topic_pair_commonness_{level}{test_suffix}.parquet"
        )
        # Upload to s3
        upload_obj(
            work_novelty_df,
            BUCKET_NAME,
            f"{filepath_document_novelty_scores}",
        )
        upload_obj(
            topic_pair_commonness_df,
            BUCKET_NAME,
            f"{filepath_topic_pair_commonness}",
        )
        if save_to_local:
            # Save to local disk
            (PROJECT_DIR / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            export_path = PROJECT_DIR / filepath_document_novelty_scores
            work_novelty_df.to_parquet(export_path, index=False)
            logging.info(f"Exported novelty scores to {export_path}")
            export_path = PROJECT_DIR / filepath_topic_pair_commonness
            topic_pair_commonness_df.to_parquet(export_path, index=False)
            logging.info(f"Exported topic pair commonness to {export_path}")


if __name__ == "__main__":
    typer.run(calculate_openalex_novelty)
