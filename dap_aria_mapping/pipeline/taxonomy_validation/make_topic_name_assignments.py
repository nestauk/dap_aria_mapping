import pandas as pd
import argparse, json, boto3, time, os, ast
from toolz import pipe
from dap_aria_mapping import logger, PROJECT_DIR, BUCKET_NAME
from functools import partial
from itertools import islice
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
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
        from revChatGPT.V1 import Chatbot
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # bot = ChatGPT(False)
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
            )
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
                    long=True,
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
                        long=False,
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
                    num_topics_file, first_parse = 0, True

                first = {
                    "chatbot1": True,
                    "chatbot2": True,
                    "chatbot3": True,
                    "chatbot4": True,
                }

                while num_topics_file < num_topics_total:
                    logger.info(f"Selecting random chatbot to use")
                    chatbot_num = np.random.randint(1, 5)
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
                                for data in chatbot.ask(random.choice(["Hi!", "Hello!", "Hey!", "Hi there!"])):
                                    response = data["message"]

                                query = (
                                    f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?"
                                    " \n\n "
                                    f"{chunk_str}"
                                    " \n\n "
                                    "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:"
                                    " \n\n"
                                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
                                    " \n\n"
                                    "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above (it will be parsed by Python's ast literal_eval method)."
                                )
                                for data in chatbot.ask(query):
                                    response = data["message"]
                            else:
                                query = ( 
                                    f" Can you do the same to the following list of additional groups: \n\n {chunk_str} \n\n" \
                                    "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:"
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
                                for data in chatbot.ask(query):
                                    response = data["message"]

                            # In case it failed to output a list the first time but not the second
                            try:
                                if not isinstance(response, list):
                                    response = ast.literal_eval(response)
                            except:
                                raise Exception("ChatGPT response is not a list.")

                            logger.info(
                                f"SUCCESS - ChatGPT response: {response}"
                            )
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
                                    long=False,
                                    n_top=args.n_top,
                                )
                            else:
                                chatgpt_names_new = {}

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
                                    "ChatGPT failed to respond. Idling for 10-12 minutes."
                                )
                                time.sleep(np.random.randint(600, 720))

                # if os.path.exists(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json"):
                #     logger.info("ChatGPT names already exist")
                #     with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "r") as f:
                #         chatgpt_names = json.load(f)
                #     num_topics_file, num_topics_total = (
                #         len(set(chatgpt_names.keys()).intersection(set(entity_names.keys()))),
                #         len(entity_names.keys())
                #     )
                #     if num_topics_file == num_topics_total:
                #         logger.info("ChatGPT names already exist for all topics. Skipping.")
                #         continue
                #     else:
                #         logger.info(f"ChatGPT names already exist for {num_topics_file} out of {num_topics_total} topics")
                #         entity_names = {k: v for k, v in entity_names.items() if k not in chatgpt_names.keys()}
                # else:
                #     chatgpt_names = defaultdict(dict)

                # for clust, names in entity_names.items():
                # for chunk in chunked(entity_names.items(), 6):
                #     tries = 0
                #     response_ok = False
                #     first = True
                #     while not response_ok:
                #         sleep_time = np.random.randint(90, 120)
                #         logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
                #         time.sleep(sleep_time)

                #         try:
                #             chunk_str = "\n\n ".join(["List " + ": \"".join(x) + "\"" for x in chunk] )
                #             if first:
                #                 query = (
                #                     f"What are the Wikipedia topics that best describe the following groups of entities (the number of times in parenthesis corresponds to how often these entities are found in the topic, and should be taken into account when making a decision)?" \
                #                     " \n\n " \
                #                     f"{chunk_str}" \
                #                     " \n\n " \
                #                     "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: (list identifier, topic name, confidence score, list of entities to discard (None if there are none)). For example:" \
                #                     # "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. Also provide a topic name for the discarded entities, if any. The structure of the answer should be a list of tuples of five elements: (list identifier, topic name, confidence score, list of entities to discard (return an empty list if there are none), discarded topic name (None if there are no entities to discard)). For example: " \
                #                     " \n\n" \
                #                     "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                #                     # "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Russian doll'], 'Russian Elements'), ('List 2', 'Cosmology', 90, ['Berlin', 'Madrid'], 'European cities')]"
                #                     " \n\n" \
                #                     "Please avoid very general topic names (such as 'Science' or 'Technology') and return only the list of tuples with the answers using the structure above." \
                #                     # "In addition, if possible try to avoid topics that are too general - such as 'Science' or 'Technology' - unless a more concrete topic reduces the confidence score significantly." \
                #                     # "Do not explain the reasoning behind the reply, nor write anything else than the answer (ie. do not write notes on why you discard entities)."
                #                 )
                #                 response = bot.ask(query)
                #             else:
                #                 query = f"{chunk_str}"
                #                 response = bot.ask(query)

                #             try:
                #                 response = ast.literal_eval(response)
                #             except Exception as e:
                #                 logger.info(f"FAILURE - ChatGPT response is not a list: {response}.")
                #                 time.sleep(np.random.randint(9, 12))
                #                 bot.ask(
                #                     "The response is not a list. Please output a list of four-item tuples as your answer, as in the following example: \n\n" \
                #                     "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]" \
                #                 )
                #                 raise Exception("ChatGPT response is not a list.")

                #             #     f"What is the Wikipedia topic that best describes the following group of entities?" \
                #             #     f"{names}" \
                #             #     "\n\n" \
                #             #     f"Please only provide the topic name that best describes the group of entities, and a " \
                #             #     f"confidence score between 0 and 100 on how sure you are about the answer." \
                #             #     "The structure of the answer should be: topic name (confidence score)." \
                #             #     "For example: 'Machine learning (100)'. In addition, try to avoid topics that are too " \
                #             #     "general, such as 'Science' or 'Technology'."
                #             # )
                #             if "Unusable response produced" in response:
                #                 logger.info(f"FAILURE - ChatGPT failed to provide an answer: {response}.")
                #                 raise Exception("Unusable response produced. Consider reducing number of requests.")
                #             else:
                #                 logger.info(f"SUCCESS - ChatGPT response: {response}")
                #                 for quadtuple in response:
                #                     list_id = quadtuple[0].split(" ")[-1]
                #                     topic = quadtuple[1]
                #                     confidence = quadtuple[2]
                #                     discard = quadtuple[3]
                #                     chatgpt_names[list_id] = {"name": topic, "confidence": confidence, "discard": discard}
                #                 # chatgpt_names[clust] = response
                #             # logger.info(f"ChatGPT response: {chatgpt_names[clust]}")
                #             with open(OUTPUT_DIR / f"class_{taxlabel}_nametype_chatgpt_top_{args.n_top}_level_{level}.json", "w") as f:
                #                 json.dump(chatgpt_names, f)
                #             bot.refresh_session()
                #             response_ok = True
                #             if first:
                #                 first = False
                #             break

                #         except Exception as e:
                #             tries += 1
                #             logger.info(f"ChatGPT failed to respond. Reason: {e}. Trying again.")
                #             time.sleep(np.random.randint(3, 6))
                #             bot.refresh_session()
                #             if tries > 3:
                #                 logger.info("ChatGPT failed to respond. Idling for 10-12 minutes.")
                #                 # chatgpt_names[clust] = "ChatGPT failed to respond."
                #                 time.sleep(np.random.randint(600, 720))

                # if args.save:
                #     logger.info("Saving dictionary as pickle")
                #     save_names(
                #         taxonomy=taxlabel,
                #         name_type="chatgpt",
                #         level=level,
                #         n_top=args.n_top,
                #         names=chatgpt_names,
                #     )

            logger.info("Finished level {}".format(str(level)))
        logger.info("Finished taxonomy {}".format(taxlabel))
