""" 
Script/pipeline to calculate novelty scores for OpenAlex papers
Uses typer to create a command line interface

Usage example:
python dap_aria_mapping/notebooks/novelty/pipeline_openalex_novelty.py --taxonomy-level 1 --test
"""
from dap_aria_mapping import logging, PROJECT_DIR
import dap_aria_mapping.getters.openalex as oa
import dap_aria_mapping.utils.novelty_utils as nu
import typer

OUTPUT_FOLDER = PROJECT_DIR / "outputs/interim"
DEFAULT_TAXONOMY_LEVEL = 1


def main(
    taxonomy_level: int = DEFAULT_TAXONOMY_LEVEL,
    test: bool = False,
    n_test_sample: int = 1000,
):
    """
    Calculate novelty scores for OpenAlex papers

    Args:
        taxonomy_level (int): Taxonomy level to use for novelty calculation
        test (bool): Whether to run in test mode (using a small sample of papers)
        n_test_sample (int): Number of papers to use for testing

    Returns:
        None (saved novelty scores to file)
    """
    # Fetch OpenAlex metadata
    works_df = oa.get_openalex_works()
    logging.info(f"Downloaded {len(works_df)} works")
    # Load topics for all papers
    topics_dict = oa.get_openalex_topics(level=taxonomy_level)
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
    work_novelty_df = nu.document_novelty(topics_df, id_column="work_id")
    # Export novelty scores
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    export_path = OUTPUT_FOLDER / f"openalex_novelty_{taxonomy_level}{test_suffix}.csv"
    work_novelty_df.to_csv(export_path, index=False)
    logging.info(f"Exported novelty scores to {export_path}")


if __name__ == "__main__":
    typer.run(main)
