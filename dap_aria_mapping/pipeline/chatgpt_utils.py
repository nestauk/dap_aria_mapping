import numpy as np
from typing import Dict, List, Any, Tuple, Sequence
from revChatGPT.V1 import Chatbot
from chatgpt_wrapper import ChatGPT, AsyncChatGPT
from dap_aria_mapping.getters.taxonomies import (
    get_topic_names,
)

import logging, random, time, ast, argparse


class revChatGPTWrapper:
    def __init__(
        self,
        first_parse: bool,
        logger: logging.Logger,
        taxlabel: str,
        level: int,
        args: argparse.Namespace,
    ):
        self.first_parse = first_parse
        self.logger = logger
        self.taxlabel = taxlabel
        self.level = level
        self.args = args

        self.first = {f"chatbot{n}": True for n in range(1, 8)}
        self.chatbots = {
            "chatbot1": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..3ZItclUdcdzvKlF_.nk4fq3XFXdrtYsirYjTdWck5QKsxCj5NzorvAYVAnUNuP0OVmigfJGmE8MRYIHGplFYBoCf7Qh_afVCCcHBvKn6DLZeL0Z_3MXMXPJ7-ot3YEDHbkGCEL-QAQiLQ-iDwkkQyDk7NTNfBsqNZSeKs6f_2uXcVPcQfJ1ucqZKrUGBNz8iKe6LNvBV5qpJln6eG7U_hFTrGAeyB-MCLqAc0F7C9hLT9mXW9PXlToA56sBhjyK3KXpbLjUjYbXfLpniTSHAPTXB1rxzo9G3z2o7kghP-qJMvzhDpN_QnwJMcnCdNf3Rtz7ETh5QJw7jHRLhCwXrcunDx4tMmvvk_pfYUzFH-aF1IMjKs67CVVfO0BwTY8xcEZRwHvjHZCo4xNHSshXDPzBJT63pKT6d1PDHUu1fK_iwgHKe75u4IvtT6dGXl4z0UV3lgtnLDEW0Nt9Ed2joyzR9Zmu7e6c5N9MtS20DC1FsK0enBbDGoM-lfP_Yjt_JMf4eJVnBn8zHLADPzR_oRj8o5BH0SCVPMfyt55hDyvl9mdVxgEnDGxISyxzrjdspGBDCPseOrt0OE0a5pKRpXU4Py1I-fvUmLvE_LgepGcmdllImx4u2HN6e8hkAU-dpvSJnpymgOqxvsLdlmEcmDhKgrNT8AkeztLWagk0vvkeNqBbkFKd03GSfL4uoJGbs4Fbrn6t6kRY9_Jn7VInFLNtTO9NAWTbU30nnOisQswAhfgmT2ZqzqX9lMhnSPMU0YC66Rd03pP5WdUeewWttw4bbv0t3ib4VgjrvOxD7b9stXeNi0Qz1vMc5G1PiyFIzeHGuWlIhOkwUHFhEYtYAS2f4wOJ7-o1jkG6oNaWbibMlZ2ckYoIzBY4PNjrCMaCB36MRquJ53n88_2Ny8-MPbvRg9GRAdQFusWqamCDWHKAbUV-A_3WOsoS4keRtHJURioopUhO7a5FKFDbBZds3KpeKRMtepdGDBZVULnsFzXWdvF4wiPnPy8d7py1YdYGcjvACsmADcjYrMkisI4rhMMspwxelM4qu6UOkX1mzehENUmjkRRc29wqp07a5o_NwE3E9jurAPOJ1_4_Ei-3jhm8BetYrXv3Vm-UpobyKdCTht1HcrTqYfrD-CfL_G036MB2PeknXE5nTnk2GHOU0GCWLUcQ9u4GnMGG9PKxULQa3Tmho3PpJzVgA0Hx6UR2HXgt2PpxH94vAzGyVgeDA9FlbXXjEK0rCrAhDNMM3IcslsQalchzEvz9yGYx06D6SFmNQ_PR6Nz1WFovZDu0loplLV8mji5MoTKbavXdfzLIHH-HnNAULmJBGkkoZNcYU5QNVYC8z--Y2HCXzdfAbj7ngctvLeRmtpihJ4LdilgwERB3lkY7jf76LGtdScEtwaXXV0SpOJoG1dw-1ZwXbE3cmLEVvBNtFEHIYMPF_2m4yC1X3ITsF_eAHsyLxH3l6eXY7RYyUVHLAKXF1twZ2AmWaf5SznepSECY9iBwCucfgnSOAhhk__HBgvuPcuSPKsVuF5vYpD5qSX8DxDPsmZIZsQOQAspe7JkXj6_voduzbQF1E6Uygs3PfPghRM_NoYSwqn9NtHU9X8_GeQbXKcx2E0tr6VKvCcQX_fG100kQIzmEuh0LUGHVBvSF6vsMcIDq8DmpBty48j7EPrkt-a4U87oHBT9jJfJrCSApPovEbJWOwiziimMhoTS5BuiXDwdP0WdMDU1tXGeJZJVffiXYBUSktaTjnQSTkxID1SSxHKZgDswcyCr3Mef8_YUnU4Bc9uZM_cvSY9gTzkcJxKsdy1s1q-lY18SKdz8POTaj9BAdL-tJRL91Ir6X1VfCeAR3YUImBArTg-FYEgPeBYIMSMr3mIK9C0Vv1YktsEnKMu2ESt5CqclQ1h7ZvoR0a_41qYmAMzWgRdqs1vfPvRJpU7t_o6eJM6N50MGvzOrW8mmlripiGSmm11wkYTzHsRGJD5ttVICoBFb-HjgoMqD2NaG-a0UwNKhd9dhOA66Nhy9A-G663-9SLux1swThSvSE-OWLMhUwCMLjiDv9dAlQdzA5Wr6MJ8uYDNT8HCTR7KOOLtH6v7-qXQ7IMqzNL1a5CQT7PAEgcrAftL9HBvVPXUOtVCcPiN6P65bZ4P9hOuRcCZZV6gNHr12pYNJPC0NRJSVD3CDqh7Svzr3asj5XWFWaKiGnzK39krHDta6vT2zVVIQm7JidgkHQb8xMU29gn2eRT-JLFC9IM1zivohx100Dm2DLqW8Mbm_BJxVc1tmiAcyimxV-PrWD3Mt0gKQm8bgO59HA_iaovPhHK9BRDydGRDJRTIxosjGqGJOUeSqsjB31TMkBdAoUja7axJROm5BDkZJe7VkEI2sMkz2UVrhbNNMESP9LGe4JiRMwHT-dKF7lZdr6HKyZZmCJNnWxK6Tshbz3IZFx3E3aBWLEHBWC8ZQvwMd7wm2ZuAPANaYRDCx2EcGp3LIK_06WmyN7_oO4HtX15NE66Xn7uzmW07MWBCxPN5Nru9d_1zpsoZzYVF.Z_OrA2bpL2cWAvCismK0FQ"
                }
            ),
            "chatbot2": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..uUTcyEPNa1P-rWO3.MjPx2y1t7diqkKzJjZMv6x44TQDM9fh82iV4pMriOAVzWAMZTuhQVF8TRocnHtKVicdF3Qp98Pq8qnewoMlm2NfwBRdlGou_CwS37PpBeEudL3OCt3_2zyNfxKMMmAFgwgpCdiQxPPQ9rbjTQo2KXLMLIFnRFE6XRzYXcrNgygb_aYBwWxvGBWoeclDKzdvQfMN1ij-6m3M0ei-4sBcAY4X3ojMHBqxIZ9FbdD9pmFS1JiTsXC-p4pSoiv7TGAZGFDtPxCbpfJCksTOONKILHEB0XuuXxSVeLoH96YJAK8deEQrOAhC2tATYHXRaLLd4L92UfxB4fspZ03yPOFI1PhJLqFhbHz13KGMBNSZHsNBMUOx_Mw9V5l_Rr3j--QFIhyo7YNai99EbMu2emZ7132IrCUBpTPXF2Nund_7GVvERXB8Jmv1IhR9k7gkwb0-BBxVzwtnK6TuR0dr4CUflJPkc2mzP7gn5DN157vIxOOtUEDPdTJpd5ZrNURNa-CNgDzuth-s3ZK_-EnUCfdVsye8ACS9IBkbdng-JXoGupgBIdScCQi7zQDy1VEpwWzt47W4CHcEAIxEzw-jTEb3jZdgVhlaZB41gXNGCFeO0ooLBm8bQaF-aCPobrza9LCBNToS36jk4_hw6qcjW4LtxSIkgkAYytV8ogOEOAZvVAppETark43HJIAixrC9ON-41eMMEV9bLtYtE2PI_3vpkv7x4ihA2Mfg0Gta16qAKdYuvnbZq-zzKfFmfoIi_vb-ZEGjyC9pFFNuzKTfFBK0BBCMYWUw0_iHKxy7-SUZ0pAPxoY5CWB9XKEKt--tDTjBZTZEOLy8adlBr1WESU03eUisz9K22ccOpTUBjCuNbkNK13szKqOUlLCbx3hZ5S4AEx-4RWchmNKhsBPpxtsbzWxSpRvtAENdnbJPQlzrNssUTLbpdUSQGs2IRaSEP_LjAFW8RVatjYgkQHYOzwfkHWort6OH9qvb80CUknA1lnssPkncxap73w3lSzQCDX8DPXFpA_B894q7rRs4lZ2VQE-1lmOik3UNEwB4rmuL301wjl57UxbaigSqY7XpRmmouiNcXqBfI8jnyRMehwM9clMWMzYiBtSFMjdbYgcSMo-Dv_Ikqtoso1fmF0vv88_iB0HCDI1PZveO6qq95rFmXDNnZ06SvmtgrAH8_w5-qgpcUj6vuAHoxb0fL4QJFqwmR6hYkHfDx1LDzFuFhuxNNrH_0-6EkM-TtKApuf8fCclYRcshndefOW5Rd6NSUNCnGouns8IBEK-3wzqdV_m2Y04_w4lHBLMeATEhLubhQPKQCGbEqwdzg755wkHhMek2U-XU7IMdwLHGladygXFNJMY57GvvXpoyvCslDbCCehIYn8Iz0z9KX_S1kcFvJjrws6DT-Hdc0ddHsgns0pFNVa0bt1Obe1AMHISCvG-HB0rQGchhXTu4xSRzbfYyNI7vfiUopbeSpoJKgpC9iA9nCtXa_A6loKLB8-Bsb3sBEeJtGlep-sVkRMQnNle3b0m_s9eQGx0mY0vGHj1gKR2XugEVpEhkrZ3lXCiOHT_Non81ApNQ9VIu9IknWN6YY-HS1uum6lcCzo6XV6GtvbW9Vv6WsL4VkFu_HXEwTt2Jl4uAgnw8GXFyhTFkITYhdpwrt2sov1txBjIBCcfAdqc3SSN1cqGlEyv3vSQUjaxnycPJ4381sReVYtWFQ_nfTVbMubHDSgMwItkNMlLPqaBibtfV_VPdh7l1i3_iqYmH7HKPsivSwxXv4Uw9lo-HUJE8ONolXxFQgMXVhOOznKwxzugvlfb95fzTUrun3lc4w3sMzcwnUg09pxZf-0pO0HGDk5Q4WiIVtPzBBMrsXC5JUPCcumjSG-TaNOFARoZyPUSe5VcOoBVA738iim-yrbsfR9UZuERGQVlQ__VqhoHNd7U-YNZ2Tq9TiYvlpkXkXo1XVbIgd5WfS7OcuXlmudsgpJbwo0jauGPVOAv41p7Eh98GBOyz-8EeQQyAXPtebzzodcsArUjoHGq1harBgXbVEtITlIYvL-imUYSoBPBhgzc0rZ6kuflGKzsgLriEu6Cyay6Z24Yy7szEOPvfcKH-WSlV5G4k9D_Kj8tELxZ6HrwLgOQzSqjHUqSXsakxZt8q9pnkaDkO4f-5KTGQvwUmwFgcUnUU7Z-YDuMoI9LedbPiyl_gmpzQ_DJXzNvK-kv3oq_5nhYbtqx1zpXbzx8kEgj7y9Qa7oPoBr2IDcatIWCqyfFkzLfy2sqH3Bt2nbpwBBvVk8vxenOXbnfrkA24U06YMRliRRHik03hrboLR0XbqSsjNU36rdde5xi4V8T67lmpUDYaqI_4x5sjO1M9e_cvpwcRXoHYo-kCSO_PvWxaniVkm_JWAmuvhToXHeg-O5K3rFGLMFc6sLWd5B1IU9-ysj1YpuzqqkYvJDaTDJgUOvhPhb466SfdFayesqpeLAgCKAPMoRxYujrN6CfdDoVRAGw.WpRwuB5eftU_SUtTzwAZ-Q"
                }
            ),
            "chatbot3": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..akAvXoMd6NOFxrUZ.wjA3XW3P2-_roEZ6k0VgRgCpK-0WzEZqcDzj0zwl9OiBsIprqQtCp5C0ufcnRiXcmFGlEPILwit4yfkAFr2x5NWi-RrGxHqI-yt_O8gRWRxWHXNUr3eMzUZZw_v9zE7SnYqMgfAiFeVZmJplgZ3iFcvJfYsoDU0uV8INIK5MSKEG1zEENG3krb6tL7-xTOEKBmZ62kWQm3aFgWTPuBYtEHWVMoOvKxiL_a2fm9byfqpZipCz1qaMgkHS5xIYjr8ZDbaBGaMPhDDNMO-2jrUNW0P4B38A1ufP-_MJ5TdwSHYmN570KntkQwNoN1mp_4qJ0v2aRLwbSrNmn3tRe3UBvKkOTLFfRYKns-nyPhZozI6L2PVYO3L_ZCX44a68J21ky77CcLHfQZ6WP0_uPTF5Omc805HJ7RpVXv1AT5WdHJwfUpLNb98NhJYTK4peWbHr_Nlh__5VpUE57WQqUvvIpeKhPXE0soqNGV049LyVQjr3Q5ieYqet9Xmh0sfV3FNPlmLp9wLWxsZuqoeUFdR1FhZ1drI3Faw5Y627TogyeJCzkvt8_nlCxYxS6j90sU46k-d9LoQekzor09_qWLFKNQsOeN8EGH7DWLcue0XmtldIae_OpHAT-L0Og8FyNQLOkEAqXxjSydc21iHdQM6L7UM86f8Uh_nsm7Voig0c2jZigg3JDY-D4WuiVcQkkDN6t9w5XrgY6WiCuV7HGOxwRTS8Jd564KgFoK524YF274tqlGkTFuUVlmRMLodVHVaXszmNVq_lFrfAZ3PN1rUPxcMbkU8-zHkesWsAqupg6CazAoquDdqJ5Dfo7z4PphW17-cKEvbMNAjzS_DfFYC2s78s8CY1rjH0u9MZokUm7tvYCQaF-RYYSJphW7DDA1W4YGv0zf5vwjPREm7UPm8Z0b-TolvnDFw2L6JEG3RDX0D20gLhjRKtd8xiQoIovqvBmYwPiXveSnAOnYcVqlZHpF1489XtosddmjDGXdCXixf9LO9Gbq4xU_8onWZS6YHNUkAe_gqDckERTrEqOfVx-U54mqj0ONlwsoy_Q_3HkkVWYJWPwthfKw2zT9VVQnrD97b3iGWNU2DR8KPhhUgqwsCMqxMprhZF7eGyEslRplZVApouHgvrAZbRXywvL96HH84BzMDQn_9g9m8b6vhdk60BO7nqvXIK8h8eZnAxbZ8ysYGs9X2pwmbZcrXzrwpEzgoldMFGju4yLMyqn8itCXO-V7Qh8vMURAkOawUaH_nfoXID_3mq7W64tzD6WJeX2CpWLyWsYalfbMgEilLwL51i7TwT7HDh0lsdM89ssYrjn3TnPFmLgm7qGst9qKIkm9atQe5CmN6IMj189V_ODAlpJJz4hwPO9fEnMvSjNYx1SNQh-2AamkJhCv8g_7PY_73Meq02EtmJwIm7wJMq5XRk6Pt50Wpv3AIfgX5iR-RvCiNNe91i94BESK0MrgKVLANcDsUJTP2o9E21mJShg4MLMlk6i0aCblfQbcnUca8iyZpTVYXygT9rqLmrUyV_uOeNRXayp_Nn9Tr2Gcj7qpxmeeYWVd1KDCSysDWiaqdY7ayzrZDLefxtIRoJtJTK3wB_9pZqk-eEjcir3YZ60nmd21n7p6d78xrwoqphUzGo_VLOGhQsuzXxkYQwwGdRrrRYYfu7o3g3R-lKtmlsNX9x_LmLO7UPQ2hyhF-v6-z_nUmq6WPNSv-Wg1W9Las-vZSOrD1cWaaoMHkx8UNW9tRizVB-6Xd3BN5-4PikAhRLlr9vrxOeZhKP6frhn3DtsXqLfEQQYWFVPuh15yESQUfpUE24G8q2yPhjsZPaEaMmo7Ccokf4aqMvoXeeO5qaPw6uIizr4D9Mxnoe0bhWskeBlGHPXD-QSqnVY6T0Lj1lKIJ9xC__xtdzW4Kx8r5djzd1_gTRF0fONsAzw1cdSIan-aSxYb04b4ADeyxq-YsSqEqxEgjne9j8O7RkJF4qg-msfXYLW1zrMuWGq_U29_Qmculp08GsI8k3mk1kDGIYiOpfEsA7953yM5sJMYy8AhnhqxxqmlvFo-M3o76lREgqe_4kTH3z5agMrFoj4898Fksf178V65laksamA_2HQGsCG7OdJrHhHOc0Pyp3AGClNkjSEvEcmyiA07eLKSv6XVJfx57Lc27y0OSiTYB_8iEhFzzHzZITWKsyNshDYzNNCBDDagTjgLy1kjaMD33S4G8sNgkNbvkpKi1qESgpfb0YaDqGFuNNDPOnrYrufTmyzoQ_7POvGp7623ZvN9f3e0Xy9BcMRduSS9d9S6qC6Cf_ECtypGaUf88pMsi1FdGPHz80JQVDJRBp_XCZx6HTtNYwFrGkSNIYTTXkpAzSNrqftlfrE_3EILVmtMr2anh6P1sKao-bPBnSydwL2dAcHCiShfGTPxQzQEU1kcgMO6fxF0SizP7FEzdAkzB6olFKNgGhPPO45ljMZrMsepKBBuPIK8qx-S3lqFce4kkJYsICvrzJZ-O9hYkNvJMZ8Zp3Hc2lfAWphMSXAqE.sjUsS6IcjhtH7-5hOWQV-Q"
                }
            ),
            "chatbot4": Chatbot(
                config={
                    "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..VnFAEmnEMF_xHeD3.9G_XSihEP628bXW_spSbC6Sn59JVpz5dyr0xNY3BCUGzhzlI_nkOAdwnCL8X1ObV9CucCmW0af_6esFG7ifvrMu4mABJNuOkIKf9PloYyaw6RkKGIPZhaZgbMzO7MMf7HotXbwRZ1TSlS9ho9BTFS9qliQLVdMSgKFDPDUKxkM7z-rYhUnJ-9XL8T0YebwhGaJstZEwCHOC9AnY1Y6QRLO1OOZnqIFpswYk-4ZFX1Jhr3iQsX2zN_-jKYdo4RkRog4Hi64wW6-tT4dxUvLSqPM_ZdS4iMp01FIRP3bh7EWtIABWtRGfptkgibGA2OYMwuxJx0WW_yC2rWSPPk7SK-2ocze1FzIs0W6E6rDkdwdS4z5nxRqS3nuuufyP3ynJmhc7Mz3ezz7W6GMiob9pMyfUDSF4lRgyyKfi3Ayd98SIzHB4QWiBWO8OpUe6GGHK1wiCSt9kuoenEKgwea7Loc7qQsjdo3AsfzduSKVPYGxJ8J8zNh0qFIxmsu9UN8g1RQCApnP-bGtFetcnTG5Vu0kpaETgf7tYjXTKnb_58AdEbsdWms-OtxPMXj1qJuCzru10B4ZrnnAY7JbzA75iRmwFXjEFchXRaO6fZK7AZ1pLwga-qZ7Wegouirc2nuB89juOrvsVUWCnS5-m2pNuviXfcwA2bdxJbyzoyMaJ7e1iYl2gj9g_Ob3Ow-PFifi-DTJvbeTowNnsTltCKpGmPz174GAhtiIZB48PRscMEEh4UfkcdVgxn5Pe_vq08bLWeODV4uxWirTKuHGUdDiULUREH1macz3h00_PZjU-avfuKCLddWSA1XNSAQY7Dlwj6RLKmmikwziaR2dcfe7mv8_d_F__D36dwy_EsZmnUfVgOxgElqEXjmguGDMS1Ffe5UiyJNzbmbVP7rKQrFQPXj_Uo5bie_czVcWXssIjoFaUx84UISNMjL_pIQ4RcGnCN6Q9w08csSUsmMYnyIohd-4AtS11cV-RDw3vRHB-aSs2sNs4pjXnDqufNJzeFAYMs1bamHG2yEMGmc55BsvtLxmOF4mLjrWqsEvhQq0mesbs5Xxar61UMtQZnuHKQusZwDaLJwdMovvgkM-tciqCC4KA9kF7HIx86ytprdRAHlbWqzJAf_nKlG5-EqOAcvA6vDBJERd4dwArqu5yys1CnhP4vXyxUkSSeH_Jo4y-mWKtCINh3a0FywVAEOq1qt_dfsEbQyU_26NS9D_5PNnqwVaWhGYvHrrsG9HITm9pub8ZlQ2ffknMlht-2VWWxddcST7PDxrgiH9oC3IzjfE2klY7tIrKsfjXKnYXisgzz9R2kwZSl_rJ1Uygi_51ymaKOZq9hmeyedfJc2rJwcEE4MUA5silzfrJvarCuOdU_dqIOiUX-Di8d3YSb9pduJi9nF5npocOUPzRUc5Ut0SKSsnT6zbsNiQG1DHkJ-4gGPA_tErSXofpk7-nzGEyFFOGGfTnBVltIxzSrKOmZk1clEUbY6GX61BD935PVr0u0cRM4zGO4DN4FyNukeZHLsFKBTJYCKdJb4j3g3Ytg2CADcxrv-Mvm_oVKp39RpYb0nJ0ZlIoPpsQp2UEh9WsYSnqGvZXbJJ7JBmmz_13QwEAtg0LYpAQPLkT57R9KrxzKt5xEAlwjkHDJeUomcUcNtdPdlSXAW2p8-hHcjnKxv4BM7y4tLnW7cO-mwx1UmGMz8lYaA5ph_5Y1EtBsPubFLw-Oqli_lWd7oeaCA_MtESKc6b2HlIK-X9fvbnsORkr3f1XPTLrU-RdbbsU26jnRXb2bu0HqhX4IsONN2HdUqe8ALL_BpVIsTr6zjQlUIHWPzsUryTEe-jhsqBEWYmGGYznykxUlFVDBJbe2uRonzV8MILfyontErFnBoWKh6tOT3_Rhu77Gr5O0PpeMom8DVFsDAaf6HYnUpbY_wcFKLSr7xrgzw8XwotUaprnxppJfF30SDtoxHitRI446Pezjpv9oLJkEGw-NVC6ZskpvBdHFknj2l3qQrvoQ5CjemGl5G460utyBlPPpmmsAksjNLPzYdMO6dkp83GQxwkgBGTkgS-ywxm7VDh-kiGAKii3QNefez7vi7fMiBslgiZLMtMH8DigS-WLHtG2jvYm4_dVh1mbiUbTudRssJa7WM1LrwXrychVF0MZvglLwbBfkvqQGfYgNydvx7ShnuUieUiEhUL4mgndOGFM-yTIBM7Iz9UMWH98zJfTI2Wwt_X4AhrYk9RDPcY5mtwJCHM4-btvU2clO9cIe_-BXMFuvMm5IlCt_NZ176L3Ox-DcC2yXZsUF1S_N2ZKsLRvSyJdn89xcb8X_ojjLINHhl2GDdkB_4kyK3Rzswpfe-qqR01SLAcxx8N-s9HV9Oqz4XWe7_JfWc_9c67t7DTzZe4DEOlx-Y4dHZqfW_gAFC038eNYCd9bPx37M27ldU1v3hXx-dgwiqYmXib_DgKgvf9ddDA767atxYWKpjNuPlCWH8OLtRZNhNj1XlzY04qdqCP4-mTuLbtXfvZvufTn73Bilh9lMIx92JZrXZUNvO5pD.X-FThIJHTY1oA5FG322-Kg"
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

    def __call__(
        self,
        chatbot_num: int,
        chatgpt_names: Dict[str, Dict[str, Any]],
        sample_entities: List[List[str]],
    ) -> Dict[str, Sequence[Tuple[str, str, int, List[str]]]]:

        chatbot = self.chatbots[f"chatbot{chatbot_num}"]
        chunk_str = "\n\n ".join(
            ["List " + ': "'.join(x) + '"' for x in sample_entities]
        )

        if self.first[f"chatbot{chatbot_num}"]:
            for data in chatbot.ask(
                random.choice(["Hi!", "Hello!", "Hey!", "Hi there!"])
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
            self.logger.info(f"FAILURE - ChatGPT response is not a list: {response}.")
            sleep_time = np.random.randint(4, 6)
            self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
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
        except Exception as e:
            raise Exception(f"ChatGPT response is not a list: {e}")

        self.logger.info(f"SUCCESS - ChatGPT response: {response}")
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
        if self.first[f"chatbot{chatbot_num}"]:
            self.first[f"chatbot{chatbot_num}"] = False

        sleep_time = np.random.randint(9, 12)
        self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

        # load back (in case other streams have updated the dictionary)
        if not self.first_parse:
            chatgpt_names_updated = get_topic_names(
                taxonomy_class=self.taxlabel,
                name_type="chatgpt",
                level=self.level,
                n_top=self.args.n_top,
            )
        else:
            chatgpt_names_updated = {}
            self.first_parse = False

        # merge any missing keys
        chatgpt_names = {**chatgpt_names, **chatgpt_names_updated}

        return chatgpt_names


class webChatGPTWrapper:
    def __init__(
        self,
        first_parse: bool,
        logger: logging.Logger,
        taxlabel: str,
        level: int,
        args: argparse.Namespace,
    ):
        self.first_parse = first_parse
        self.logger = logger
        self.taxlabel = taxlabel
        self.level = level
        self.args = args

        self.bot = ChatGPT(False)
        self.first = True

    def __call__(
        self,
        sample_entities: List[List[str]],
        chatgpt_names: Dict[str, Sequence[Tuple[str, str, int, List[str]]]],
        tries: int,
    ) -> Dict[str, Sequence[Tuple[str, str, int, List[str]]]]:

        chunk_str = "\n\n ".join(
            ["List " + ': "'.join(x) + '"' for x in sample_entities]
        )

        if self.first:
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
            response = self.bot.ask(query)
        else:
            query = (
                f"Can you do the same to the following list of additional groups: \n\n {chunk_str} \n\n"
                "Please only provide the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, please provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:"
                "\n\n"
                "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
            )
            response = self.bot.ask(query)

        # Attempt to convert to list, if exception, try making it explicit
        try:
            response_str = ast.literal_eval(response[1])
        except Exception as e:
            self.logger.info(
                f"FAILURE - ChatGPT response is not a list: {response[1]}. Reason: {response[2]}"
            )
            sleep_time = np.random.randint(40, 60)
            self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            if tries > 2:
                query = (
                    f"Your response is not a list with the requested structure. Remember that I only want the topic name that best describes the group of entities, and a confidence score between 0 and 100 on how sure you are about the answer. If confidence is not high, also provide a list of entities that, if discarded, would help identify a topic. The structure of the answer should be a list of tuples of four elements: [(list identifier, topic name, confidence score, list of entities to discard (None if there are none)), ... ]. For example:"
                    "\n\n"
                    "[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioska', 'Madrid'])]"
                    " \n\n Please try again. Only return the list in the required structure. \n\n"
                )
                response = self.bot.ask(query)

        # In case it failed to output a list the first time but not the second
        try:
            response_str = ast.literal_eval(response[1])
        except Exception as e:
            raise Exception(f"ChatGPT response is not a list: {e}")

        # if past the exception, assume successful response
        self.logger.info(f"SUCCESS - ChatGPT response: {response_str}")
        for quadtuple in response_str:
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
        if self.first:
            self.first = False

        sleep_time = np.random.randint(60, 90)
        self.logger.info(f"Routine idling - Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

        # load back (in case other streams have updated the dictionary)
        if not self.first_parse:
            chatgpt_names_updated = get_topic_names(
                taxonomy_class=self.taxlabel,
                name_type="chatgpt",
                level=self.level,
                n_top=self.args.n_top,
            )
        else:
            chatgpt_names_updated = {}
            self.first_parse = False

        # merge any missing keys
        chatgpt_names = {**chatgpt_names, **chatgpt_names_updated}

        return chatgpt_names
