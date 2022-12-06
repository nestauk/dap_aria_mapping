from google.oauth2.service_account import Credentials
from google.cloud import bigquery

from dap_aria_mapping import PROJECT_DIR, BUCKET_NAME, logger
from nesta_ds_utils.loading_saving.S3 import download_file
import os


def est_conn() -> bigquery.Client:
    """Instantiate Google BigQuery client to query data.

    Assumes service account key is saved as
        PROJECT_DIR/credentials/patents-key.json. If credentials
        path not set in your global environment, set credentials
        path. If credentials file not saved locally, download
        credentials json locally then set path.

    Returns instantiated bigquery client with passed credentials.
    """
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_file = PROJECT_DIR / "credentials/patents-key.json"

    if not google_creds:
        if not credentials_file.is_file():
            logger.info("credentials file not saved locally, downloading file...")
            download_file(
                path_from="credentials/patents-key.json",
                bucket=BUCKET_NAME,
                path_to=str(credentials_file),
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)

    credentials = Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(credentials=credentials)

    return client
