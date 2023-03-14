"""This script makes topic name assignments for a series of taxonomy levels.
    It can be run from the command line with the following arguments:
        - taxonomy: The type of taxonomy to use. Can be a single taxonomy or a sequence
            of taxonomies, and accepts 'cooccur', 'centroids' or 'imbalanced' as tags.
        - name_type: The type of names to use. Can be 'entity', 'journal', or both.
        - level: The level of the taxonomy to use. Can be a single level or a sequence
        - n_top: The number of top elements to show. In the case of 'chatgpt', this is the
            number of entities to use to label the topic.
        - n_articles: The number of articles to use to count entities in a topic.
            -1 defaults to all.
        - save: Whether to save the topic names to S3.
        - show_counts: Whether to show the counts of the entities in each topic. Used
            in name_type 'chatgpt' as part of the query.

Raises:
    Exception: If the taxonomy is not one of 'cooccur', 'semantic' or 'semantic_kmeans'.

Returns:
    json: A json file containing the topic names, saved to S3.
"""
import pandas as pd
import argparse, json, boto3, time, os, ast
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from functools import partial
from itertools import islice
from revChatGPT.V1 import Chatbot
from dap_aria_mapping.getters.taxonomies import (
    get_cooccurrence_taxonomy,
    get_semantic_taxonomy,
    get_topic_names,
)
from dap_aria_mapping.getters.openalex import get_openalex_works, get_openalex_entities
from dap_aria_mapping.utils.entity_selection import get_sample, filter_entities
from dap_aria_mapping.utils.topic_names import *

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interim" / "topic_names"


def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(islice(it, size))
        if not p:
            break
        yield p


def save_names(
    taxonomy: str, name_type: str, level: int, n_top: int, names: Dict
) -> None:
    """Save the topic names for a given taxonomy level.

    Args:
        taxonomy (str): A string representing the taxonomy class.
            Can be 'cooccur', 'semantic' or 'semantic_kmeans'.
        name_type (str): A string representing the topic names type.
        level (int): The level of the taxonomy to use.
        n_top (int): The number of top elements to show.
        names (Dict): A dictionary of the topic names.
    """
    names = {str(k): v for k, v in names.items()}
    title = "outputs/topic_names/class_{}_nametype_{}_top_{}_level_{}.json".format(
        taxonomy, name_type, n_top, str(level)
    )

    s3 = boto3.client("s3")
    s3.put_object(
        Body=json.dumps(names),
        Bucket=BUCKET_NAME,
        Key=title,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make histograms of a given taxonomy level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--taxonomy",
        nargs="+",
        help=(
            "The type of taxonomy to use. Can be a single taxonomy or a sequence \
            of taxonomies, and accepts 'cooccur', 'centroids' or 'imbalanced' as tags."
        ),
        required=True,
    )

    parser.add_argument(
        "--name_type",
        nargs="+",
        help="The type of names to use. Can be 'entity', 'journal', or both.",
        default="entity",
    )

    parser.add_argument(
        "--n_top",
        type=int,
        help="The number of top elements to show. Defaults to 3.",
        default=3,
    )

    parser.add_argument(
        "--levels", nargs="+", help="The levels of the taxonomy to use.", required=True
    )

    parser.add_argument(
        "--n_articles",
        type=int,
        help="The number of articles to use. If -1, all articles are used.",
    )

    parser.add_argument("--save", help="Whether to save the plot.", action="store_true")

    parser.add_argument(
        "--show_count",
        help="Whether to show counts for entity names.",
        action="store_true",
    )

    args = parser.parse_args()
    bucket = boto3.resource("s3").Bucket(BUCKET_NAME)

    logger.info("Loading data - taxonomy")
    taxonomies = []
    if "cooccur" in args.taxonomy:
        cooccur_taxonomy = get_cooccurrence_taxonomy()
        taxonomies.append(["cooccur", cooccur_taxonomy])
    if "centroids" in args.taxonomy:
        semantic_centroids_taxonomy = get_semantic_taxonomy("centroids")
        taxonomies.append(["centroids", semantic_centroids_taxonomy])
    if "imbalanced" in args.taxonomy:
        semantic_kmeans_taxonomy = get_semantic_taxonomy("imbalanced")
        taxonomies.append(["imbalanced", semantic_kmeans_taxonomy])

    if any(["entity" in args.name_type, "journal" in args.name_type]):
        logger.info("Loading data - works")
        oa_works = get_openalex_works()

        logger.info("Loading data - entities")
        oa_entities = pipe(
            get_openalex_entities(),
            partial(get_sample, score_threshold=80, num_articles=args.n_articles),
            partial(
                filter_entities, min_freq=10, max_freq=1_000_000, method="absolute"
            ),
        )

        logger.info("Building dictionary - journal to entities")
        journal_entities = get_journal_entities(oa_works, oa_entities)

    if "chatgpt" in args.name_type:

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chatbots = {
            "chatbot1": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..heSMPbaDFqCLUD6K.GAPPWlYIay-_b0GgIMYb6Ui5rAGxpSa4BOevkxNvilLSq95iDgB8cY0VUy1QqWRUPcCBQepufAsDAn3CvrbVTpfFWtB8kCaLQwL2MO_cXlZPowzy3YHcHz1_12vkreggkL9IVKdNaMlOlne9AVPZNLAhxDhu6Nerlw_ZcCEbeglhonDWnDTPSDbf_HzZT2SwhhKUFWg3iPlj7GHX6mX_UmaAv2nNOs2xTX34rcfOFjYHCO48WToDipy8Z2qTJdxnwJgnRK4V_uH-ucXKFc0yE4H612P064vsMpmEEaVwsqM1EZIFrcTLxmmIqg3MF2w77jx5XHBN7mMMPdVY9aV-h8E0nqFJLkmUJhaWWZIa0vlnOaEHLmllMgDLiHN3ur5-75gLFbHxuSgcOkrLMso4BCrSx22pBkyCLUMYu3D5N9q2zD3vn_nSVE52st1Lg5P64wfKNJIJk1GH1U1BmiALyatrVpHOMHBeNZ5ZpZRKEvZSXQILts-Ej1wYLZ35greMUy72oZTsuLTMLR2a_7py4-8pm2TvKJgWxgnBzbtfcli5-CZhYsjnV_AlyVh7KbZrMhw8_QuCHB5zP7xsUmq0tVi3Em7obGP4VRVJaIN4sxA6ZV3gqdFAmLtBITBSJh3jB74Mdl1XTXSz2Cvu33pvF8qlKxZVXcgGvb1HXuqfQQlAnpwNj_CthV0PBERhm7dR5nTaiRc9i9MGdwBfpga32zEVR_ZHxkp5DFs3UuS0IwN4s6HRMDy5bM8ytKupVZ6_Q4OVoZKOce40bK4OsVkHGzsivjhICNnjMJKh8Oqarf1vgP2xqWKF9rRoDcP_EEsvbjM_JsWxbvjOaEVDcqEwVOJ_suPOtEy2s96yi_ZRMOPLp52TtNdzev5_SZvznlJCnzbLM0JDV9RNarsHpcHzBu84IY-ydzcXRlz80SBWHkSNcTd9tMcTkNQG4f3yJIK-ufhfoWXme4fdBHhVICRBoe868hGYvkojRPRA3PXOisHOeHp7ERwgWQdNOErBBvly3M0B87QZT4ceGZKLDW1REJRbsXTpn0lb8ZI_OHVfvvizIHA8KSQyUE2euCtzdT2J5UJZUiO90ksui5QJKcUf2HGSL9WYPsTp8pOJNMD7qbmCAbp1u8dLhSxlvXmNsoo2EGLSGnPvPv33HWC3M7xnS7Nb-pC3RqBMqFPUI7vPj9EyE9HYIs2D72M0o-_AJZzEGvzcAJAo9nG59v8aFzWOMuWwrglNBAUh1wJt_mX3AVKwEFU_cQdTVAmXb4thUUFcbZ-_CI2b67XtZxi9N5t9BnIxhozbF2tC9gIk04scLdHTa5nJXsAOyjeGQwghIgC8CEzbjP7cOEG0ukWr0ykSjY3pQlLyDWJK37C57RFE9zQ4hgym__8INJ3ytS59qEDjnj7r8ZPh-_HlRFUFBJGrchndTHqwpB19q_ycb2p5-aiRXdYLu-hIEqxpNQbyPzw5Q32wUwqRw3SscZcstpDJfNMN_tLOABEZwheL14Cjm3noZS9S97mlyJeb0wwHR3gfuAcdsiHWN95Ri9Urlh13M-iRS-HiYQ8UJPapB4jkWwFpJCPCpqw7Pxq50KDLSJUMtQVGFd6_OYNG1vOuXj929M-YEjjluQQY8diFv-_i3bUTuVD8jnhvi-zgjMLM9BDgTsMnDwVFVb0U1o284paM_fJVxN7SxKe7Rl7hvqwelYXYi2GOwL10YbL0v3jjj5gqKf_6qNXIKNH1tVWUYQRvFlu8lkUF3usBZqBh1RtIBHSemV1a0XyrX5drUVzsvXN4NjR6EdmVnEOkAZO4VE5KLXkajf8KTm_vWOwxpnyKz-rZY-Mq7wr3X5k9TYgsXqaPceEyIiQBKLeWQYeDqRnQqFOL0GkDLwKdf9tb7IbdltWVef3FL28tbAUk_sMpRdpVyDQUGMLwzKLuDgw9cOtgUJ4PBtsCAgFaGxJF3JZQb2eE1vnCfQr08VD_1oUc3mDdjnGTsFWmLObcqpEoG141jhuoiw64EsCSVwq6SeWDIqUlr_XGh2DC-yTXF0GAkDv79Iq-xw6752d95FwxUKMT159aRDx1MR2HYtOZ2GyrpavFNWiaD-mbzKnRiHEd5RWLX0jeIpNCRWw6cINJ8gjU7gL0_wrWIexDAu7PwZPhAT5blSNyQl4JB1UfBOLlIyTgu2Cl_9R9a7rXC1tkmiGNNeDmkI6gkE1nH1wam5c0sr5Qnh5ad64OhAynkrKbk90bCUWGaRc3NWDsmCbKei0PU8xMRyAxle_CefeiskS1zMfZeMhrbXFfsuIRk3GHadGu0RWL00tvR7L-DYkHyFIxq-DGKZiV38oIH72u95k6ZXG2LUqSfp6IulYF0Ogo4w6-vbruBfTDFEvNQFkiHmmc0CDb-QiWlQ4fzc6sUj9HjcpMPXay_PeA5JZDOlu1zLUBsA-82CYEUQvX1PUHfE-vQuvr_r7edfeU6bvUOfy7mdpYuab8Zg3vT7XhCvIZpsMZYGPIVohjlIn1.lt2c4obxI6XeRfLV8BKH6g"
                }
            ),
            "chatbot2": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..JEuSN_u8rQL9tEme.JDObm-5YGQ2GccMGTIJRZjh-4sD4RI5NtgeRX9G1oeAStVBS2rU7A0Ov5fcyxWnDDz214tnuVq1h2dJfOq6o7ulhZyIUfIU2zlPY1aYFZ3UhZwIMscoQebL3OLmYToLFWMsWWkawy1ROvp7BxGhQOvaM5q3tppYxJZKubyrwQKE5sC79Xs8FEGhZu_KYn5SpHxrsggrDX0qb6qCWhGTubqSBlAkjfwAg6LHAVlx-huSGkvHJMMFmqPwJ4KJ8xTnbSLBPqXkWyaQPiNPNTIH5pyk88_GFwRGzZv1T0Eyen0R765ZcKStzR49MJgnvLZ5rb0qxDjXcWon3WTbeS29_zaTIIEWdAxpGOGbSzq7TeaqRaEzcPWm1F1cBVfnuuv3usNBTKPkQJOpNGkyelIApk2Du8eicMTTVJK25x7UxBDQ8Lphu1wOGAOFP7NuraddDYvGVxL2IEube4m6sC8b9-HJkKPJVvUGE-exsGgjemRBOo_wnbzSLdHv3KuIKOLA7iJhVlYZ8F9cZUR0DRu2vy_JmIl7BPJYG4QJ7I3IeIvQQtiXJDmH_lQQxZbkdJrQpfkg00hQacAAPABBPElY-yQR2-nF3gRzvjUTB5y-GUQGiZHloIuyw2DV-T2THHtw-4Wh_DV-RAIHatYccvqMGop6XdW4DHvO1NLcH3U4LDej80QPjk-kVEb1opsR1MfPb0u4FYLipJ-h1rcDpKdVgnsawZDRA7eQ-Qmz4cYBt8lfIsEe1MWcvPuF5U_q41IdQ7aqyFThtFsKkxhip3q766LGXV959rGkHHEOyYHMyRCrCAXk3Ix2p9NnJDX0bAxRHOyZLKNvnoiv2jzf33QkGvOhmZcZ1uoVlKlt3wrOXq4mdhXkASHGnX2DaK9m9jmjAjXX2y4TE10cCmAObZ2FreV3TbNYU8wpQ_nBA94Qiu4iSqJD3UMTakLuBhdGzVyMzhGs_l-a0goWpe3ISV_m2PG-VOT9AYUmT17eON8wuU_4ltNyogEG5ggljedfvPCAeydb-E1N-UTts7w05ufLZKipzpGYvVVrAn6P6OeZ5fJWQtQdAy6MUOxSVP4v-SZchv4rNqZN0fHidU_x1cHz9RAy5wPw5bAgXJvATAGouxqWq44l8H5tJqK5Wn07DBHtKtELdCIforcUpFNEKQ2Mb7nIfANSHo9kwGDYrW_9TjC354VFp_ga7iglBw1G9SZ8E0TwHvfH_BCVhRXKnXspcNCCzeupzXzUzxA1lCnBO-5h1gZsTAaokg8G3DQLO0gvq4LbSGq_aFqEBu7ZBHjvypLXRpPdQCUs2d1474wnb3ZiPesGKoxTvL8q89SIOFmCT4LOlPU0zcRtHBdWW4WbMObGx6qyFiIRPgurtwqoDcCWyzDIFuHRYCQkey6abZXEPTfBAT5qaEC-i172yYbIvcgT8MhFnqoQRNT3SHRofHamU56U4MdOgVH3h1fg1almf5Svy_hZv3wYCvICXJZ3vS8agFpQPezk-ibDXeDWtQe5X4OwRFfF3O_SijuPPn1wMpyN7DVjFzgLSE07ILTQYQ4NKiC8khMLinIKVFVsDLExqE9NKGKkI-dnWBEFygJ6BTu9OcRXl4RugqQiowhUrlY84IzzO-y4FyGyhGe8iMmTIFdt04KroHjB-hkGa7hm-qeRG2iMEMI4XXoQhr0jJW-OFfycOIJ9wFuFp0S09not7hgtpX9B3j49a6ipmzvjRx8bxu8tZDi_BvZ4fsgTlrH0WsYWEhBP-14tPtRy07qpEH6farMI8HG5-mQsJG_5pdBhw_h8LT8Fyyyd_wmEyYTCcyEsI4jKZ3hNmFBnPmfCjVdUfYlEEEqKkkiRWo6rjw_qDSzxo1mm1JDx7_5FypSXDQ0FDuaRnCC7gWaSmDKX1r0AwdeSrfbwTNb2iU2UTjesvwL32UV0Z4VZYsBLrlb_-O4h4r0CR2MnBt0WtL2OMjgyaYSbdQKiWMhWvZw8Tt5dcOaijNxe56nJzTqAnIQys7kxPNCyk_WDUZddXc9JNcastLAFlVHcz5SMhVCxYwDOJuWF_vo4LWBAD3EpSioxp8nR7poROeSirEUeN_cFF_BM-5DURwYiuoJwHh5zMGKPyfqemdVJY40VXKN5Sjibgg_mYXubV0kITGprCOPFTSXWlZ35gk-FeaDnA9O67c81ZHbl_bBRzJ3Wy-UOkvW8X8HbuCSClxPORnxNsedMdKVPLe-PC6nPm2Be1sgx6MS2Z-jI_c33Fpt1S01kjAs6ZobqePtQZUktckozi9-Xm6NNb5AWKlNJwoPlS_HE1Nd_EswlfLkC-DR0yi7AxQzo5Q_U3LMERbTczy1w2Swx1l8aS-_lRa7ZqY5cNTtw2MsUn9oscVNKyLzQFMGwByGk6YKw0YaJNSLcqCTUDvR1gOQ7kC40pygCiQtLG-ONHIfwoWxvXs9jkXiWze1uRoG7xnEnbqd80Pa8LwOjZcdGgigJRYe97BmqRIH2mygbPg9NGsDPRMdiy6pWUUgtkE0wtiHdVARUo.hqgaUKweWYlGuphlTOC9Tw"
                }
            ),
            "chatbot3": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..93fQNkXAgCvD1pet.DnF-Rs0s9t6p4b7dFV0MzJhD5aDprpu19R5ojG260B9pl_KF4V6YY9QRKAuvPolUMfRTqW40JVXL0Vb1CcipBGQI2ffxxMFHr8a2yTJj6tXF3c8TTBKSqPjFC33SQAdXI-8lmEmofdcGz38PiBKAzFSmrHspjDxHLy8PzllGllk6gv61LYPr5ZgVco0pXlESU4Ebi9NDcWgJMz8RnYQ2EpdxvBrVkhCUyKd-BXvunTbPTMU72ENMz-0IPVtlX1dbKVOQS2To5e2d2rvi8TWsw00BxFVetXQXlF-6275sM5HkZ9wq6CDn8gjaoVzWiyNJI4YJmfOgZdh7UhFkPEnYVgwRtCtIKQNPtrzbw7DAoYOB2s_JK8EPcWMH-IaJOTmWLpTNNBXGbrpgs8Fydt5LiXaVNrOdIt_mP0O3k8y5JtmT_ochOtRol9CWwFkM5OFqdxb-wSEfTa9tQ84Q-QzDArOKyCz9CrDVMsRlpva1Y6DIJgXbmoXFvhcJJwdQ3PAN6_msFedVg7Z4rNiufm774kIu27n-sJhEdJR_5YEqqhhwJheQFJ70V2NXJDdTYZHK7PCi6ZmROMSlRQ1hJ6KvRf1csdoAjlExI8__cLap78l_R1CSRuk4vulucnDJ_ScQlU2jMlASr0XJ9WeGr8snNzSd_pMF1bfACEfd0T439rh8OP9f1ei5UhsPophsFvtrxmlfnTaoKS4ExlSSmTAlKo_kFj37ioNSKUPNtquoZs3n76tlO-v2p2C9t8AoBNPsE-1Nx-BE4jkEWuRJVY2N-MpJ6oA5V3SOBdK_r-qIdmft8xC-L47hzGNM_Hq0DRhD-B_1rPrONmbNRbLvbTfTOxur8E94vXP-CFn5PRFHl9vqCP1SbFWE5LUEp093W_VPRIfoXDzqjEKII61ey87B6bvSLq3jMl3Ak-ehA1SvpgZC9YxT6Mw3AIVIG0Kpy1R30_dF4ydxkDXA4RDy-w6hk3kW16OZ59rgdfIzYyj3fwvzy67SuduhldEFwt36NNNG32HsIjW-OF017fsp7xcUHeYqUNfA8hS8Nau9wSNbUdBTrwLCZBbEziRLbEMXCTrq-Yu8Fa0xO3Yi4Qmb5MJ5jH44h_x6TzPQZXi9V_I-ya1BYJf3f_m6S-FYYTo_XgA5TY6pZtONVV0V6S22Mx7r99CW47gTGXX8dFKTIFmPaTZIOmdftcL1Oq5KwsJW1EzMS5ffYqeFBTRaKea4KPrjCTYTwc2VTkCB9oQLmC8oPof5aTDNQ0eigmDVfxpWLD_ZyTS960WdN6ZYJsfke1wR8Ecn9t0uH822_0wOw3un6SaICp8AcjSpOJ4ffXi8AZgTuwP9zsYxozhFqsk70VMyAepUkSLJherrEczs3G1QtjP-8jDVnR5v1_SthZTQ4E5M3wOJxuTj5XK57emUJyekYgN5d-Ej3CeYggnTRHUv61KgiIldwq_g0qKQqRa9BAQctvQHtIbv9gi2Rsi6vmO6KbVtBxT4gqyOcPvNxPS-c6g_z5saSaKabjKcn9LYiBWUi9woWIewHsVi41kR6w_Zd9Q9rE_QN2zNDHqS44lNUQ5yGAvl74aZaLiCZpZOlCna2fiH8W_O4wlAJYeXh1NjjwdWr5tBriCvkML4Qwst6nFVkLgkRGKCHz9wxgZ60uMWjzpbutg08MiVeUzRtwiJwzGPSFLbMq-rK5oWusuTW4RnPHaUOJmC2O6EAvgWd-0PXSune8sWsPz_JUIa3jypW64ipbY0RNrURtEGwUpJlcuwdB28HDk4ULFZo_S1CrQmbaSF1TkxISTzxmERw5v9hguYJM1cqzArvV26DeRcTB7_A5sWE7cdT75yYcH1IZUXJsWmjolX96NwJa_zEsMcbFat02Ovgwuc7p_7qB4Aly_33d42sRf3WitvmwwqozMJgySoGjgd8up1DC4OW3Z_Ujf8wtVuEFLT9Y3FASFZUI6OxEnkbUHwgdMZzY4fY4cLdNPiwUB_XTCgewHqZItstGNrXwB7Hif6mfd-4guhwbyyz8t3S9wMUt5YVvMy6Xp7SEAJv5hee-sRjk9ae8qNh5tPgz7J0ihAYCf2Gqlj5IH00uhLpbnvrj6yMblKFoo8gqmhe3tR3ZNEosBYYutzefVMlLTi8o9wyDm_RCrMZRyoGiRGcQCudXiHNflVE6TZvVi1AVBi3rAC0Iq-Qc-iayuVyuZU_F7Y5OOHR1bh34ruG_Zm7Tx1nswTsKyFA3Ag0al8YwpxP3U2g6-hc-RcHL_rUiEFdRiAnhIu4aEbu_fcZXSA_RZTRYFATSmYfw7QN81k_Vfm-jndzquCkIRaf85ZA5HyruUQkjIBPMa_f3_Q-M2YmtUuhyXyb40My0QZuRj52sCq9RLoQ0diZGFXJzatpBgvH_WmMH829-tKRrFGK_UJYUemxkUuHoLX1YkcPo51uXaatJnCQz5SfkiMq8eP0fdZRBF94ZfS0g6e9ZSSUQaqGT5vSWNV3Pu8bdIEwqDToqgxXTkT2enyyOCFhNkskr3wRJUzWoHf074.gjiB_upsuvuNti8VgFW3Kw"
                }
            ),
            "chatbot4": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..jJyP87UXxdkNDds-.UeOGJeWy2oHxZP6ww1h6MDevQn1bDSRbbTnUu3Uznz-C4lHbEaNELBy69eKDpGp51W0YBfFmyf9BJSYFUVKT3lA1YufY2uYLdPRtTXLZXRsPRP7aaNnIeA3hWWZon9QRkRtMqvLkNhsGDjDF5_FaxpAhc1juiB14pRrnQXARUITrt724xxpDQW5quf15ueSqBwzo5WCg0XA51NN4CGN1WvIGM3uZS7iQWlvE0j9buNzwCNHYEMDjTo0dpqPGnxUuZYBou1eXK-ynm2l4WQ7K-YotI1naBQx92OrG1eKW1VxcOgnnOkKMro2jUCEjoXxHqh5ShfbwJ6TPadclWqo9aTcpszdUx3ZhEDuklr7KFFweWAMG2kuxqtehZcgKpW8GEKao56QoA104NUUXzQEzQP5RPIpb9xOBtvHqdS_YJlBlbMtY7d996TGTPTfDXMsKwfw1-ZGg5ss4geq1nRoRcFM1koVIeyIACgSjIH7Vh6tdvZ4flK57ZpNrOSYZHs7X26035ir04B2-nv0FqqrpY6GuVG58klIMiM7ctjh9ndFW-PzTCNiKNh5opuaQXcQKWGYGO6hsqea8IOk2cwD6CZAQ2LcW2p3bXJV7VgvDYN2bobEVfGbFGwpRXgI3pq6f-GRNh394OwgN32yLFiq0C2hW5tU-uHAAOM-bfjA8nQv_gioVLK1_XAJnIjE1nZcp5SsyO6FpPm6f0Pv42zMF2mBEsIGTRkB1An0Aarpz0voVpf_zwdHyqtbqVqxA6wZFfWBE_FeBpF8-ii7J9NH4uaGlGl_J8VxJpICI74wcn-9mrVTrhWT7usEptZKxVm3SMZOoKMK6EvioQA9E4OViTXNb5rDa2nCA-Q5w3u5Y9ze9oEDZADc-bcSMeo9iDo-6A404WVIWjLsscgWQ0GL0vQM7ModpCB9OE-XARDRul0DFAUY9rGrxVW-p1Ejx4Ar2cG3BOtv1YBfG5SRiBrcHR64105toItnrzbOfcv6sMtS3O33BUhXFuKLf9ihHm4sNvPeihVICtg6_ukU0iUlfXh7LYnlP7Ds33WTA6zishXoDH4UqUJwhTUrb9NFggbg83zxUwr0ans7ftfQ6CC5oL1x64qbfQa1nIKpEZgRk0Ngh_YqcS0v0bhz8KzQK_ifwi_wREt19kkykgL0UGQg9FfAxPH1DRDZ8B0SHmS4eb0Cc_CdT7gV9IBV0F3yE2pAnV0fV3gXiG-ZSeKNtt8GFrd2mmxzNj5uyZ7Bcz6M5ASY01HWCYIMUbvt89aSw756KDcMAXdTx3QsBQPu_l8YKL_L17_JPpZ0jeZLZyTr6psds9W_FPtSJBeEpMCfooYczM_TiR30Ii2vG_275YRx_QRwN0nVnagMVBQ_7ohf93LqumE-CsVwapIt-VVsAkP545e0vS03fz1J3On4Hm1MAovXeQkbC_l40n7PRLQdGQXDL9gvU88zx3cqFs4ih11MSv3FDw1ka0arySNHzkCqjgkjXBi1QVYTmMzvys0AYG14MHdjkdVBOU9DK6cvjFoHWGLrGpNHwTm12yOVA5fDCfuJEs_YQt_u-zaiUUCPrG3618poE3lUXbc1trOVxuZ4vtl5Plfak-OKGqRoYgIuX2n4SV0o5aY5q0vB2t0rbK8FwEJV5rh2g1y9Dm0b7WWQnic7ffoAEo26ljsPVVDQ7IntTRM91lgzVrquh__dqTdiWZu3oqbOdfwX3NNOLjinQL-aNjmzX3MgGHgIP1QYlHz46VLYzm4dJGn9nlBq8FrPOcmyyTB-BaRfbNZ3zDoutXtASPh8zKVQhfQEpWaw0O--BgGN43Sct2QKn0rT6MBEoPxbIdFB1Wva4ExcZTL7PW_Ery4yy5by8NiM43pogcGAdRf5zOGkCy1lVsMx2LoAdxLcb9K8hT4BSANW8A_-lOUj_1640_BQ0Z4xWXTnU1KrPT_TUvnIMIzvkF2q4duc0g_xcklmm1GPtvuH02rNNmm2aZUDRqa4v3EhKyLJ2Ph9g0k2jhfUJcwQdm8YDH3PhRM_1cNiI0fgixS1RDZqqES-QcNyxaQwTT7GwsyFyAXmBquMyCeCGNgzBrGqXRosx-ybVojShjZz1M93FC7FAyoe0pIYoISovsKKjtYjkZcFPn8j94QJeZRik3EBo24lETqM1ZIJ77GPByBLZ8o4XWnNs-VI6Xe-p7dEhHYXenXLFxjI9_4QnBA2bB06qIDHL6f_f04zanGgErkluGeEkYXhws3_xBQ__nT-rBup8KQ3Z_C8ZwJmyawyeHQ9EML5c-wSP7wJ988d2snzrrtrctsmJRVyF2an__4UYU5d3PU2WSaSU0-m9oqJTSjBb_rwxvh402vO0K9ZeV1t65u8bMTBEm75mePQT7JZ4BWrREkgfCArW5eV9upCsNdxf3QayR-Cv2fp7ayUJVYJ8mvjaBQlY1RGBrMCUATSrlfHxjiFXP4OgScauA2XdFdQzsMD9VfTYTV3sNTgsexhqk9-kXUqgPeZhdnNZZXL7H-X5LIZ33i5ov9ubm6cwHLM0tPatAtzxmh_n8JJX.iu7-LEzBzaF1OLP1Z58cPw"
                }
            ),
            "chatbot5": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..jL7lCZu9HYBzo11o.s8IBg4gNBFXwu0cbvKkOIu9YtjIrNKL9Nqfl7-uRLLumtuWPgJ3ANGVi4rb0zyrFbofCRErwYZBGOcZzAlh446Nag4EnnJNBLpd01bsrT7dfVbvCGjB7VmCtgH-9lbfbDgq9k779jkIYdAq2MlNOzlMBR9bZPMTgSUJY8xQ0GNvbgTmtRSa9C8csdH1q95BKMWVbadTTnCb8cJHDT-Pf7ebYLHmHEjmXyV2PrNU_VG5ktr9ZD0_fNNOwG3gAGVaeXUORK8tZOvORQFRoHEe0zxYZ3hLoxWiKZGvpbHwqtthGutaACYoZUVUy0CNBLjFXltvtPoJHvR5qOf4tSlCVzAzjDuf9jTlrqAlDJgEBYHaUhEziZA94q4ePdTTvp9yfWIM0orTAcH9TC0KK0EkzoPQHw0OliknwfXRkOVI4LmNmk5uQpMXUYgWl5z7KORDsHruBiG9SvhPvBZx-ezRkcIm40CN0o4Kk5hMXg8FkEHRrrcXwhPTX6wroui3kGH3p8tD6g5lBk-XQuXcI-SkW8HyA2noeStH_DlWOySLQA3kPrcMPGJgiGKJDwW7QgviTjbAPIwWNoqnFa1elNKIVvRPm-i4K_afegDXuJDfuqpfX5rNN9-we4VTnP2F6a721zVI0_Qiy3PuHuABnDEOuy7evmJ9hMJPo9-SQwF4MIaeyZIuVp5H8tWOlZj9lW8Tb2Rf_x2YymGOpWdVcQUQGu3MUkOH0Nx7E-2UxxkyHoy5Ij9wlI-qZDWsPmY_Kf9Z56iz7HWS0Cl2hAbtRzflCk5FQwkgeCDRpahRvjN0CwUG2kG5_UnFZcwqqSw8SmIQpB1AN_-KZoFKSDIB5nhikS-TYa_JYeHpV_uTnzmKF7IqTDp4d3BFW1x6FZWSkt_e8VkW9UZuxncQqu_kPiOq7BIbzh7y4XtgnIGuOM1VdgIpOO5KZehs4x_M73SpRiwWziWhLNrO7M0_E7pmAHWSS6QrTWY2OVVYTShofSMLllrw6VZS8o44_Rh-LLM_bPbKCVkS7_gDjzYe_Q41bbKtWorsm1CL-rMaoYGkAldkJ5UDsae5WXA5NCtwvLHnq__T4VazS5ADfVIc0P3l8j81HiuRPc2a-F6WaEwdUP0rCSk2tiUGtSTCHez_f_xAWdNmPgv-dj5p3L4NL65PRNekPI7x1ANw388ZTamLQ7Rux815uZqUmcB7-JuhFEuBiSZLb0pkxcTMpLWzNnLvFvqh53xX1pmKTrSl560RouZKaIfpelN1oEhzADUMBfmvQWlvEu5gjw8XNRHVOfUtPxKFUh9ScmuWfsBji2xlQzDNvi6k95Owc9pHqNHEZlIMyIpNvl1P7JnzJ1viMvfzo3ojXAgfssQDuCDGQkhyfuVQv983UNAhTKzXHZ0VPDK1xMujOjxV_7ih_LxaAlbbDgfc5T9ou8MXh0fFpEEypx0l4sVjlm88UeAjefv2ImxrW7r7E4S6PMa53yyiWjvSdP--u3w-MRvGb5hq1Al_VVKn3w6KcO4PgzQcX2yBzcTxuoKBaHtcd-peSoXYk44j89eNtlEnUrwZK5lcbY4YO5K16DwtfWXNPahrZysdc5d_7uirwRor67IE5j3bjexAihZ8ih9l1aYoXa_etSAOFtacAY1BeIk5ltiL6iFR3avgDkGUqtYW11ydBzjo7q7oceQpFfT_3yByjbdsE3N0v72DWv4kvFiutu6KfsLQrwcKWkLNFLsRqRLpaQ2GhxVzi2_QpoVCWq53GgQ7QRZCDZm40uT8Fu_wYDVQ08VEaUJZbQVDNPD4Nizu5aQJqmMTdjcgZhJneI6RVyWO3GtL3u0c76Tu-HKtHkTI8WoQQ_yOYNxJaOCl8nv5dA-VfIoSZGGG8ANfpC0XbMtO6_MuQwI4_RLw8POg4zBpWulZv_d3a_sv4rtJRObp6iuEnoaDa1F9Lk1odI1a0BMrKR_e_pwItlmRPpfZZBS9InA86E0GTfCCp3pkON3Ow331idaX7w3TCh87lW7dbhPyySx8YJ7aS8-NEB8SIPJ7mQePw6Kci5RQ0hmD31zS1yo0jkIMoUC2Sg9cLGk4OdiH0_vwIQMNKOtOJhsSre6V9jTaCzZ-Y__k6AiFkalaRfeBFu853NbqN96dHw5kagzAN2kp7UGpKaetISiiD31EehiwWrC6VbFT-uEmKAnN19usywf00yowTOSjTVKjbkwyNR_pwhfIfV7uel3_QGiKIAalkMPG3p47J20rXykwJy3d007fChq7tBbX8_J-aH9QIKUeIKbYFTcdw88kECTxAmyGruSC4RrSGZAEakwrd3dhNnT53_R1AD7s7_Ed4g3amv1vRu1Ap-aR9yqsFLyuDa1EVMcom1VkdxVloTtbaym7kA-HaJSi7FcwqIX-tiHt6unWF1EltatEnMBtEOgWpXPuN5lfrBfIJkhBvV12YwG8qcI99bXGCxMAE_AWw5IDo43CzACjowWRZJ_V_ILM8S03Lh3GD387y8-yW3OgCCckAtSR98TTt9piAvsWXMg4lfThf2MRki51Y08M8UxL-gRLMbwT_.j7O9SR3tgFQ7nfkCO_TO0g"
                }
            ),
            "chatbot6": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0.._tD5vlUyPPg0134X.uBoJnL0ExNv25fK4ie96Lg7quuQGDaTtkqxU8lR7g_AUE9GWKOzUjOaD2YGFG3Wl6D_2HNbKgY_vLOxc7DSWeWQa28drL8A9K_Kv8PPlgSmO8JRq3GDtZB5vPKpJZpYsxnw8zUdrhB7bk0ifUyZ8g9S7motah_6RafkA3icp_y_f4vadHCjx7x8ZYAB6wO-_8y3Q788_n9SZKa1KD6fjQuBWQSlulHnlR_TNMBkHw3scF4mJkDDc6IoYHxgVTqU2x-jq1K2e2dzZDDa_eFQFG6TvIhgHH3ulgrtBDJ5kpafsBGFkZRjGCM_Q3YyS6ajPxp-dHZUo0lcapfFDeXiy6Bvuq-bHONh9Jzo7VhzIz3Ad5dqJTsK_QUbAJznywUsS13ClcR5k7Rbn8Q329IseMkXx5d7usmJ1R5tBRTip1iINBPqnXD5Hp3xF4YtwU1tUl9BJK3emw4mdPyNUALnyA--iyreOPhte1N0AgnTGhyp6glT704qVG4kg2f4fAr5OQ9WBpZ0lti4z0UQ_5wsTOCGwECh0LQdDoqVnBGcnhDdiXooVJM8J-H8ogfxoaG-Z0urFNfzJlMQa4P5QxkRaUIS5GWqe1Cjal0JbDTtWxgWZCA3t454wB5UCB4hhDOwNZsuk-n-eSDd9VjNHe0-4Cq2K8bfSeO3FuzmE4Kn9FFp_fkiQLylv4PH5JA59BPUSvP_qdfy41EqdTlqXZgoU9HepehbC5Zbdrre1GUpH4vUbIlGdJlLGltmFwJBOA9imVXYlUWeHYXeLd4hVRQqToP85a9kyWYPYkB-9LS6At3ZkS6988SEmLeN99hluuToT67MT2a8WSzxfVXzFTZUXYUadOcr3nHJXmabwXjUGdpLoGUE498fx_DfdhZ8C2D_CnQK9aE4Ga5tI20CZPAZXuilq430E3L3flZiJObD9Jr4DAYGpROWW2keQRYUzhnZqONp_6cciDH6XyWgrzN0MGI1ZxzrKgZV1pzs6JmnCVsfyCXYjQaaVQnOCygCFUlpynrvo1-YitbHAD-p35xib2LGbkJQxuiTNQxTnj1Nbnacv6_c5yQ8suN56YQdLlcYaGdJ2h62laIX9z_G9oAruyq0G89lEURCZjOLWWDsOz6fCeevyvPwGzAPNTPxumMb6NWksZEgifJnaFlo5PXx3KRqjKjne00Up9JwNH4Au7gFuc_iHHRTyeL6G1Ey4fKbIRe3F6miYytsq0adV2jM6FXd6MIWFo0X0zgIzWhLyUf7kR72s8j5r7Y_tux_mEVvPJmCPOzIHd-E-YzCxYwHhbFVG8l2NN0umFhnzPA1mg6h6hA6nJ_dZ8ej1HoE7e6zhpFOijMoXOzQa2a21wTa21WLtGFdmAmGqqnSMJUfplzPDgaNmHTN7VEV9hNuUG-Uhe2k4raVUluoHUr8JKVhrCNHuEFRjgiuDI569Lvh_a_HY9y9lXDLHdkj61xuByzSf7bnOI3JVEp8OmQhLkhktcRxH9nHGJw8MoVa9p3G0ub87peiwpgCSe24CWkxhLTWloXmo78DVN2gyUGrwnE1c7o5OZnKGh5-wUijHXvcOAvTK-jPMH4MzFIk9nYbDGZTS70amRp9GueHZ707pX6upyx7hn-OOZieF_fMsn_wdOvOoJ-9OUcL6mX1eYbhF1DBU_jpQnYUF_vWcmKcTKo8vJCqb9aTjeM7j2H0Vdr9cM3O-4ybIiE20TUYonncboWZlFdG59NtER_Dx77A8YzHKkcEu68zTWMg-OTHn1Ushsqb1wB_pDnWZB2mRk25dk7LIm5JoMexRa-Jdpk88_PitSX5N-ctHeza--OT-cRNn3Sw63D1mmpN4bgZ0H9NRKCjVzc4gdHXKFbCN8lu8ITtKNz0xpIkxekoD6r49m7pp3msfhVl4O3CJWvoOdmtssO9zSCd55_W5XSDRmAtS5QdkBmoanNOt817eCltNZxyRch1FqV41hq_Oz9BDJmqKCDOZrM2p-oByOlZClTQVNI2Bne8qC75vuPpeAby13NFXYYWyzMKc-GVuyRQ2-dZ2va7bOvzacEONQS45KUpXP1ZxWxjAUf5FoDlGP9qELrgI03zm-35baTzxcERvGR06vH2HpqHTRk8uPrXvC_fEy6FfuUp2QNc8-uMRhBzp9Y3idHMTKF1WdQGpdZfzp3LUYliAa0uHqhG06uaOu3T1Uwb5w9e5woXN8ShKXIpmT8WsHOrU-vZjHt6r7jfk7AEf83YWhmFVHei84F_d7NqzWwonB7cWNJ_2lsTNXTJeQKTRzUeP02G7hzRjbpz2eW7mmTg9rs8_nK3iogwuZB5k4d5U3u3mbKlSPq8y4-SmvvBpPlUyG1ovR8UnMTqzDD2JSYOM7tsSdKh7zm1rpxdOfJ04OI5WWy2mKvB2dw5YImDqE9r_ZAp8k_8O53yRH-SgIc27E0rUiz7oFGt3kmVgrAaHfGAweJzgcaBHetd7N8l26jHOnSeien8n6zQoHOrflguwyX8Z3u9fxSOCKF5UjQ.LDCvJwH5y8U00Niy5F-TKQ"
                }
            ),
            "chatbot7": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..z_Z_oVk3m9NmX-4r.y8H4uJtrcKL-lagGUZ98MIiGWsvLWQm3QB8RnSE5xBPISrkKGYPb2KyIVJXWBiS_xNRR0sBAf9CWBtU1vXMje6jl41qbZoevOMvNExbLyVgKktsdmqw_162MbzrKLSBr8nE4P_D_uH_wpK4geQjkQ_6ybFFUGUkm8FBs8H2ZRHejl2FpVL062zPQCPdk5PUGSpIJ_tUj1wra3_8U8dVUsiD9d-qfCp1CJZ1T45NiS8UBg-mVJS5qpi0VGS15K1WTpzBMnCwR4OmcKXA9VyTgyeAz-8sibcL5DrjAupzbCjMZJcoig5UyAFFnNs-aIL5lha6-ns0b212NHdJHWDWCIlvYuHN10WqnC2xwQBUIlqdulrcgm1wJuT7pm-FHsX048GNCVPqm5qVtfLVATIKa2VZgzj2FrU9mcqiSDQN2vLb98YWqZ77nyEoGsmaanUe0VCBWo5R9Y-hUyLlUty99ftcDWmuMhJ0EL9LlM7FWVjaKW7F6JgWz6DMOSi1q6jjwtwIZiZRfcrngEqvrl9jGC9JcpgSrUsGLUvTnmGc2Y7NK3XnDDN3m789FTapmhQsYR5VRJCdWATv7glH937xcfD3ZVlrjfByjpUxSHZQrbRO2smTYmfzhp40AVYddN8d-xUTcUGHLivcA4KP6dlQU4A7cRq3ThhsEHEv-7EQg_E5vzQ3dAJFqpKKdP_GBrNmOlFmaUkOktKGfgFemEMw0NRnSt5uiaVkskFii0yPM6DAUWUIQ8E9gtiDAZ1r2bf7fa6DPQnJ7s8uwf2_YlAgMDYJ4am65EHZvulYy2wwJnvEFHWKon0qwgUtZqZRTPdMnyydpl0NnscIeSZXYQDjUgKGEWaatxJUFS5Cv2xPN_ZRAjObQIDHgqSyNwICGkR-LgB4sR_qOaf-hLbMZBKI-nh627GUzBtvlhMJxa7lHZNqTEiTgMgHYuf-pyEiKP2eBJhK0HIfmN-aGEfdADAw1rWdn-uRnSOsOLNvSrQHN7LR4rSDkLaaN7v6fKmH6NpFrhdY4x5dERTOg1N5JXU7UsWCf8-EVjZo_Ppa1kFr7nP6R_oTedMGt7l1T8bN5ZS5OtikHQOhQrAuVsWO9NWFyqNikAnubwg6B_JL9oWE2PN0_mV0GaCQomt6DRNDJY7j4wN-uGwBaePok6JUK8oC6Vve9HiKGK2gyWgJY0Ja9ITu-ORVcoKElfNHx3xrFXfx24vsZDmlhUw3bdwVw_fMeDiJW4v-a2FXGIgdSp4IXCON-U-yD82saDi3BveiwSQXPbIxd4dWOvW0r_yB2In3UWUcoeIjhpelHm6D9x8FdbijRthhHQx0JdlayihBkGfyIek_ePi1YULYwTM0b8kLP0epoP5ZwVbBCycK29oYIOqH9rPf7HaHo-eORR-Q1YHNuYCFcF3FR9Ex0xKf8wO7shocn7Do_yYXIAsfh30xqo3T8Jhi0NXMRKB9WHNS_wfeWM9PSVyAU21p-0YKfXUVA3XDK1xxm_t22kuqoaTUn-f04TVmPd6Z1HEpjGnOsJ4FSlCL4t8BuTAo8a9xh-rz6myMe_PpYu13Tii5EvWW4RFddggR_yciLVJJetc5SHTL-Uy54FsTLi8M5048Idb704S-SjofrMtmbA56_vafa-n2tWV50A3n7UVlo8JrkqCu2LOBFxfQVJpXzwrGyR-h66jBnVAbXDbMjsdnFzxnkmbKxBURWL945B3pyv3YTy1xntM3BPYUq1Xu6E_INX9fdOtvj0ktT1HvLLxM_rOAzOXzthWK4GI93vBUUKmv4qIH_uHagoA4e0U99h_swO5Yw_CidOaihzhwFuro7gBi-9mmiVilYLJNcBhMZ8YMyHWGEKixy2rSqBRzcUJ1DCg3Xr-PE1pdTdFAZzgfSsaMbGhNNYutC7EyScsLP2LNt2xwcPSYorQ35IozrniejJgJD2EJsoctQWSQmSiXDEQMlZnHzrGZ5ndT0xar_nSmik3hiCHJM9qurq3yGvygpi4W27VG1_NPFQdc53DdSVMFltScfpE9Rk_3LydlDuTLlKysxCGfzUnJAbsmy30SMx9pfYax3s1vH39U69fd9_e8xz9O-f1Nw6MM8AuRhn2wGrjMrIb9f0krRsAy2swSeOG9EKq5HRE1jV0MqfzA9ICCtVOfQM_9BO2yZtIVAYuadIvL5e8P6cebZTUndrURN-4aToe-dPtyPvpWXpVZ6_IJMbkoKobSgkV8FpsxK4q41YmTB5hX3j5eqY4bO7jjVMN47tC5S6MZmZoczMvv2sqC3ott-MYAlAotjY1K_WF2HhQnFfQ9Yyn3m-IGihNeU9-LlBmOQRJXGFOJHu9QVgmH52eidqGN5VFQZyVEi36p3uARvvuT_UhsBn_ztN0imptTV8YhDXF3ymg_gxKbb4ort6yQZ1m_WxC3Q8RS4wnJXh2zPsNfFFuKwSYPZbKiyo9s05TrSFuRaIYpgQaaTNGLzDsz7wyecHBUnnpMzjXd-qyWT_KHqOQggHjlmTe-XyTMyCxJAnIgGraxPTl8PalX3ZSOZQaURtb1m.XnmKtXDGhjf_VrHj-kxKlg"
                }
            ),
        }

    for taxlabel, taxonomy in taxonomies:
        logger.info("Starting taxonomy {}".format(taxlabel))
        for level in [int(x) for x in args.levels]:
            logger.info("Starting level {}".format(str(level)))
            logger.info("Building dictionary - entity to journals")
            if any(["entity" in args.name_type, "journal" in args.name_type]):
                clust_counts = get_level_entity_counts(
                    journal_entities, taxonomy, level=level
                )

            if "entity" in args.name_type:
                logger.info("Building dictionary - entity to names")
                names = get_cluster_entity_names(
                    clust_counts, num_entities=args.n_top, show_count=args.show_count
                )
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type="entity",
                        level=level,
                        n_top=args.n_top,
                        names=names,
                    )

            if "journal" in args.name_type:
                logger.info("Building dictionary - journal to names")
                cluster_journal_ranks = get_cluster_journal_ranks(
                    clust_counts, journal_entities, output="relative"
                )
                names = get_cluster_journal_names(
                    cluster_journal_ranks, num_names=args.n_top
                )
                if args.save:
                    logger.info("Saving dictionary as pickle")
                    save_names(
                        taxonomy=taxlabel,
                        name_type="journal",
                        level=level,
                        n_top=args.n_top,
                        names=names,
                    )

            if "chatgpt" in args.name_type:
                entity_names = get_topic_names(
                    taxonomy_class=taxlabel,
                    name_type="entity",
                    level=level,
                    n_top=args.n_top,
                )

                files_with_name = [
                    file.key
                    for file in bucket.objects.filter(
                        Prefix=f"outputs/topic_names/class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json"
                    )
                ]

                if len(files_with_name) > 0:
                    logger.info("ChatGPT names already exist")
                    chatgpt_names = get_topic_names(
                        taxonomy_class=taxlabel,
                        name_type="chatgpt",
                        level=level,
                        n_top=args.n_top,
                    )

                    num_topics_file, num_topics_total = (
                        len(
                            set(chatgpt_names.keys()).intersection(
                                set(entity_names.keys())
                            )
                        ),
                        len(entity_names.keys()),
                    )

                    if num_topics_file == num_topics_total:
                        logger.info(
                            "ChatGPT names already exist for all topics. Skipping."
                        )
                        continue
                    else:
                        logger.info(
                            f"ChatGPT names already exist for {num_topics_file} out of {num_topics_total} topics"
                        )
                        entity_names = {
                            k: v
                            for k, v in entity_names.items()
                            if k not in chatgpt_names.keys()
                        }
                    first_parse = False

                else:
                    logger.info("ChatGPT names do not exist")
                    chatgpt_names = defaultdict(dict)
                    num_topics_file, num_topics_total, first_parse = (
                        0,
                        len(entity_names.keys()),
                        True,
                    )

                first = {f"chatbot{n}": True for n in range(1, 8)}

                while num_topics_file < num_topics_total:
                    logger.info(f"Selecting random chatbot to use")
                    chatbot_num = np.random.randint(1, 8)
                    chatbot = chatbots[f"chatbot{chatbot_num}"]

                    logger.info(f"Starting batch of topics")
                    random_sample = random.sample(list(entity_names.keys()), 6)
                    sample_entities = [
                        (topic, entity_names[topic]) for topic in random_sample
                    ]

                    tries, response_ok = 0, False
                    while not response_ok:
                        logger.info("Asking ChatGPT")
                        time.sleep(np.random.uniform(1, 3))
                        try:
                            chunk_str = "\n\n ".join(
                                ["List " + ': "'.join(x) + '"' for x in sample_entities]
                            )

                            if first[f"chatbot{chatbot_num}"]:
                                for data in chatbot.ask(
                                    random.choice(
                                        ["Hi!", "Hello!", "Hey!", "Hi there!"]
                                    )
                                ):
                                    response = data["message"]

                                query = (
                                    f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?"
                                    " \n\n "
                                    f"{chunk_str}"
                                    " \n\n "
                                    "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:"
                                    " \n\n"
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
                                    " \n\n"
                                    "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above (it will be parsed by Python's ast literal_e method)."
                                )
                                for data in chatbot.ask(query):
                                    response = data["message"]
                            else:
                                query = (
                                    f"Can you do the same to the following list of additional groups: \n\n {chunk_str} \n\n"
                                    "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:"
                                    "\n\n"
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
                                )
                                for data in chatbot.ask(query):
                                    response = data["message"]

                            # Attempt to convert to list, if exception, try making it explicit
                            try:
                                response = ast.literal_eval(response)
                            except Exception as e:
                                logger.info(
                                    f"FAILURE - ChatGPT response is not a list: {response}."
                                )
                                sleep_time = np.random.randint(4, 6)
                                logger.info(
                                    f"Routine idling - Sleeping for {sleep_time} seconds"
                                )
                                time.sleep(sleep_time)
                                query = (
                                    f"Your response is not a list with the requested structure. Remember that I only want the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, also provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:"
                                    "\n\n"
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
                                    " \n\n Please try again. \n\n"
                                )
                                for data in chatbot.ask(query):
                                    response = data["message"]

                            # In case it failed to output a list the first time but not the second
                            try:
                                if not isinstance(response, list):
                                    response = ast.literal_eval(response)
                            except:
                                raise Exception("ChatGPT response is not a list.")

                            logger.info(f"SUCCESS - ChatGPT response: {response}")
                            for quadtuple in response:
                                list_id = quadtuple[0].split(" ")[-1]
                                topic = quadtuple[1]
                                confidence = quadtuple[2]
                                discard = quadtuple[3]
                                chatgpt_names[list_id] = {
                                    "name": topic,
                                    "confidence": confidence,
                                    "discard": discard,
                                }

                            # Accept response, refresh session, set to skip problem description
                            response_ok = True
                            if first[f"chatbot{chatbot_num}"]:
                                first[f"chatbot{chatbot_num}"] = False

                            # Update number of topics with names & pending groups
                            num_topics_file = len(chatgpt_names.keys())
                            entity_names = {
                                k: v
                                for k, v in entity_names.items()
                                if k not in chatgpt_names.keys()
                            }

                            sleep_time = np.random.randint(9, 12)
                            logger.info(
                                f"Routine idling - Sleeping for {sleep_time} seconds"
                            )
                            time.sleep(sleep_time)

                            # load back (in case other streams have updated the dictionary)
                            if not first_parse:
                                chatgpt_names_new = get_topic_names(
                                    taxonomy_class=taxlabel,
                                    name_type="chatgpt",
                                    level=level,
                                    n_top=args.n_top,
                                )
                            else:
                                chatgpt_names_new = {}
                                first_parse = False

                            # merge any missing keys
                            chatgpt_names = {**chatgpt_names, **chatgpt_names_new}

                            if args.save:
                                logger.info("Saving dictionary as json")
                                save_names(
                                    taxonomy=taxlabel,
                                    name_type="chatgpt",
                                    level=level,
                                    n_top=args.n_top,
                                    names=chatgpt_names,
                                )
                            break

                        except Exception as e:
                            tries += 1
                            logger.info(
                                f"ChatGPT failed to respond. Reason: {e}. Trying again."
                            )
                            time.sleep(np.random.randint(6, 12))
                            if tries > 3:
                                logger.info(
                                    "ChatGPT failed to respond. Idling for 30-60 seconds."
                                )
                                time.sleep(np.random.randint(30, 60))
                                break

            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
