from dap_aria_mapping.getters.novelty import get_openalex_topic_pair_commonness
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping import BUCKET_NAME, logger
from nesta_ds_utils.loading_saving.S3 import upload_obj
import pandas as pd
import io
import boto3

if __name__ == "__main__":
    combined_df = pd.DataFrame()
    for level in [1, 2, 3]:
        logger.info("Creating table for level {}".format(level))
        commonness = get_openalex_topic_pair_commonness(level)
        commonness["level"] = level

        topic_name_data = get_topic_names(
            "cooccur", "chatgpt", level, n_top=35, postproc=True)
        topic_names = {k: v["name"] for k, v in topic_name_data.items()}

        commonness["topic_1_name"] = commonness["topic_1"].map(topic_names)

        commonness["topic_2_name"] = commonness["topic_2"].map(topic_names)

        commonness.drop(["topic_1", "topic_2"], axis=1, inplace=True)

        combined_df = combined_df.append(commonness)

    logger.info("Uploading file to S3")
    buffer = io.BytesIO()
    combined_df.to_parquet(buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, BUCKET_NAME,
                      "outputs/app_data/horizon_scanner/heatmap.parquet")
