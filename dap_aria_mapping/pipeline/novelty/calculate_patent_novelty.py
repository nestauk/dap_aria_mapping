""" 
Script to calculate novelty scores for patents
Uses typer to create a command line interface

Usage examples:

All levels, full dataset:
python dap_aria_mapping/pipeline/novelty/calculate_patent_novelty.py

One level, test dataset:
python dap_aria_mapping/pipeline/novelty/calculate_patent_novelty.py --taxonomy-level 1 --test
"""
from dap_aria_mapping import logging, PROJECT_DIR, BUCKET_NAME
from dap_aria_mapping.getters import patents
import dap_aria_mapping.utils.novelty_utils as nu
from nesta_ds_utils.loading_saving.S3 import upload_obj
import typer

OUTPUT_DIR = "outputs/novelty"


def calculate_patent_novelty(
    taxonomy_level: int = 0,
    test: bool = False,
    n_test_sample: int = 100,
    upload_to_s3: bool = True,    
    save_to_local: bool = False,
):
    """
    Calculates novelty scores for patents and uploads them on s3

    Args:
        taxonomy_level (int, optional): Taxonomy level to use for novelty calculation. Must be between 1 and 5.
            If set to 0, uses all levels. Defaults to 0.
        test (bool, optional): Whether to run in test mode (using a small sample of papers). Defaults to False.
        n_test_sample (int, optional): Number of patents to use for testing. Defaults to 100.
        upload_to_s3 (bool, optional): Whether to upload the novelty scores to s3. Defaults to True.
        save_to_local (bool, optional): Whether to also save the novelty scores to local disk. Defaults to False.

    Returns:
        None (saved novelty scores to file)
    """
    # Fetch patent metadata
    patents_df = patents.get_patents()
    logging.info(f"Downloaded {len(patents_df)} works")
    # Decide on the taxonomy levels to use
    if taxonomy_level == 0:
        levels = list(range(1, 6))
    else:
        levels = [taxonomy_level]
    # Loop over taxonomy levels
    for level in levels:
        logging.info(f"Processing taxonomy level {level}")
        # Load topics for all papers
        topics_dict = patents.get_patent_topics(level=level)
        topics_df = nu.preprocess_topics_dict(
            topics_dict,
            patents_df.assign(priority_year=lambda df: df.priority_date.dt.year),
            "publication_number",
            "priority_year",
        )
        logging.info(
            f"Using {len(topics_df)} patents with sufficient data for novelty calculation"
        )
        # Subsample for testing
        test_suffix = ""
        if test:
            topics_df = topics_df.sample(n_test_sample)
            test_suffix = f"_test_n{n_test_sample}"
            logging.info(f"TEST MODE: Using {len(topics_df)} patents for testing")
        # Calculate novelty scores
        patent_novelty_df, topic_pair_commonness_df = nu.document_novelty(
            topics_df, id_column="publication_number"
        )
        # Export novelty scores
        filepath_document_novelty_scores = (
            OUTPUT_DIR + f"/patent_novelty_{level}{test_suffix}.parquet"
        )
        filepath_topic_pair_commonness = (
            OUTPUT_DIR + f"/patent_topic_pair_commonness_{level}{test_suffix}.parquet"
        )
        if upload_to_s3:
            # Upload to s3
            upload_obj(
                patent_novelty_df,
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
            patent_novelty_df.to_parquet(export_path, index=False)
            logging.info(f"Exported novelty scores to {export_path}")
            export_path = PROJECT_DIR / filepath_topic_pair_commonness
            topic_pair_commonness_df.to_parquet(export_path, index=False)
            logging.info(f"Exported topic pair commonness to {export_path}")


if __name__ == "__main__":
    typer.run(calculate_patent_novelty)
