"""Frozen settings of the paper-facing CGSR evaluation protocol.

Values only, no logic. Every number here was copied from the frozen research
records. These values are used for guards and documentation; changing them
does not change the model's behaviour.
"""

# --------------------------------------------------------------------------
# Base-predictor code: the upstream ReCLIP++ repository (Wang & Kang).
# Not redistributed here; see REPRODUCIBILITY.md and scripts/setup_upstream.*.
# --------------------------------------------------------------------------
BASE_CODE_REPO = "https://github.com/dogehhh/ReCLIP"
BASE_CODE_COMMIT = "46298007fcfa75b0ca41c3a9a8420b2dfa800475"

# SHA-256 of the base-code files imported at run time, at BASE_CODE_COMMIT.
# Text files are hashed after CRLF -> LF normalisation (robust to
# core.autocrlf); binary files are hashed as-is.
BASE_CODE_TEXT_SHA256 = {
    "model/model.py": "4a66e9023165b2b8571cca8cca9f413258b37c5e55459a34dabd26738369add4",
    "utils/preprocess.py": "c6f2df5626c86385ff57d232ee17407fc82080eba6781f68ae803e3beffff137",
    "config/configs.py": "46e3cfaedf4a038a155cddc2898c3860649fe80fd05d6f6d2b8da8d2d55e5d18",
}
BASE_CODE_BINARY_SHA256 = {
    "utils/cls_token.pt": "153b2bdc5fa225e78adaea34675db71fae8f6c4e74437af3987ed0c4197745ab",
}

# --------------------------------------------------------------------------
# Fixed candidate generator (SFP + DTLR) settings used for every paper-facing row.
# These are the defaults of the candidate module; no YAML overrides them.
# Keys are attribute names on the constructed model.
# --------------------------------------------------------------------------
CANDIDATE_ATTRIBUTES = {
    "sfp_enable": True,
    "sfp_topk": 800,
    "sfp_top_fraction": -1.0,
    "sfp_logit_beta": 0.55,
    "sfp_conf_thd": 0.97,
    "sfp_conf_scale": 10.0,
    "sfp_entropy_gate": False,
    "sfp_margin_enable": False,
    "sfp_margin_hard_enable": False,
    "sfp_proxy_enable": True,
    "sfp_proxy_lambda": 2.0,
    "sfp_proxy_conf_thd": 0.95,
    "sfp_proxy_kernel": 5,
    "sfp_cpsfp_update": True,
    "sfp_dtlr_enable": True,
    "sfp_dtlr_beta": 1.20,
    "sfp_dtlr_sigma_s": 70.0,
    "sfp_dtlr_sigma_s_rel": -1.0,
    "sfp_dtlr_sigma_r": 1.50,
    "sfp_dtlr_num_iter": 1,
    "sfp_dtlr_boundary_only": False,
    "sfp_dtlr_structure_protect_enable": True,
    "sfp_dtlr_structure_gain_thd": 0.0,
    "sfp_dtlr_structure_classes": (4, 8, 10),
    "sfp_dtlr_class_beta_enable": False,
}

# CGSR operator (literals in cgsr/verifier.py; listed here for documentation).
CGSR_OPERATOR = {
    "window": "5x5, centre included, zero padding, -inf affinity outside the grid",
    "affinity": "dot product of unit-norm projected CLIP patch features",
    "temperature": 1.0,
    "gate": "strict q_prop > q_base",
    "selection": "whole logit vector (Lr if accepted, else L0)",
    "resolution": "native token grid",
}

# --------------------------------------------------------------------------
# Datasets (paper-facing protocol). PD = class-presence threshold of the
# shared evaluator; it is applied identically to Base, Candidate and CGSR.
# --------------------------------------------------------------------------
DATASETS = {
    "voc": {
        "config": "configs/voc/voc_cgsr.yaml",
        "config_name": "voc",
        "num_classes": 20,
        "num_val_images": 1449,
        "reduce_zero_label": True,
        "pd": 1.0,
        "text_embedding": "voc_ViT16_clip_text.pth",
        "checkpoint": "voc.pth",
    },
    "context59": {
        "config": "configs/context59/context59_cgsr.yaml",
        "config_name": "context",
        "num_classes": 59,
        "num_val_images": 5105,
        "reduce_zero_label": True,
        "pd": 0.85,
        "text_embedding": "context_ViT16_clip_text_alpha.pth",
        "checkpoint": "context.pth",
    },
    "ade20k": {
        "config": "configs/ade20k/ade20k_cgsr.yaml",
        "config_name": "ade",
        "num_classes": 150,
        "num_val_images": 2000,
        "reduce_zero_label": True,
        "pd": 0.5,
        "text_embedding": "ade_ViT16_clip_text.pth",
        "checkpoint": "ade.pth",
    },
    "coco_stuff27": {
        "config": "configs/coco_stuff27/coco_stuff27_cgsr.yaml",
        "config_name": "stuff",
        "num_classes": 27,
        "num_val_images": 5000,
        "reduce_zero_label": False,
        "pd": 0.95,
        "text_embedding": "stuff27_ViT16_clip_text.pth",
        "checkpoint": "coco.pth",
    },
    "cityscapes": {
        "config": "configs/cityscapes/cityscapes_cgsr.yaml",
        "config_name": "cityscapes",
        "num_classes": 19,
        "num_val_images": 500,
        "reduce_zero_label": False,
        "pd": 0.5,
        "text_embedding": "city_ViT16_clip_text.pth",
        "checkpoint": "city.pth",
    },
}

# Base predictor weights: the author-released ReCLIP++ checkpoints
# (verified SHA-256 of the files used).
CHECKPOINT_SHA256 = {
    "voc.pth": "f5e68096ed6ed364ab395c22b116932bc37b5c6d3af8901169b07a6f77faf23b",
    "context.pth": "21d78dc19ff3b8d3527e83da65c04cd9e178ac507712e89d57fc7f2288ba2f97",
    "ade.pth": "8b05b8c5f7d93bcb52dac134ea1f58a313421f47474702d3f2e80101e43380d9",
    "coco.pth": "2c5b55de933df0a7d1a01716c2c55a07e9a954127a0f35934a7667dc24519371",
    "city.pth": "a7daa5dad6e2f4132e3b9c24fdd8027ef22155b08d1f15295953aa383f7d3399",
}

# CLIP ViT-B/16 text embeddings used by the paper-facing runs.
TEXT_EMBEDDING_SHA256 = {
    "voc_ViT16_clip_text.pth": "561365cf5569d24334d4a3da5014813fcee2a91d90d1bfcf97ebfded0e654207",
    "context_ViT16_clip_text_alpha.pth": "ae450fd92d5a63a5ab5b091fdfb06f355c70e9d60c940f4911c8ecef747e956a",
    "ade_ViT16_clip_text.pth": "5410010b4df79fb87bad93cc7f6db607b8399164406821d78f4a8d32e564578e",
    "stuff27_ViT16_clip_text.pth": "34044aa7c3c1a16f2b04e59727fa5b40ad80188d7f8a505431783c6c31c81f7f",
    "city_ViT16_clip_text.pth": "28b83c5465091a51884fee1b1d6e211a0c928d2b612318d08562dfc9ae7d1169",
}
# SHA-256 of the tensor content (C-contiguous float32 bytes of the [C, 512]
# tensor). Re-saving an identical tensor changes the file hash but not this one.
TEXT_EMBEDDING_CONTENT_SHA256 = {
    "voc_ViT16_clip_text.pth": "e995afc286eba6eb6c5374feaa742cd3f36ad355ecaa5e50658ba4c8e4d14167",
    "context_ViT16_clip_text_alpha.pth": "ca7313e88704c91c71ca9971f71e6f0e60520461e926509dccd4bd360d1e87f8",
    "ade_ViT16_clip_text.pth": "0d0085aa432c3707867f4bce9849cd1d7418bb2385c0f768044ac22767f54436",
    "stuff27_ViT16_clip_text.pth": "1f66022c128e66c6eb1a1d94ce7b58abe0f22d59e33b7bac2647442c92e8ae3c",
    "city_ViT16_clip_text.pth": "cdc2712b780d01081da6cd5132c022d56a111d550d3a6b68fd02ce28614479b3",
}

# --------------------------------------------------------------------------
# Paper-facing results: mIoU as fractions in [0, 1], author checkpoints,
# full validation sets. Order: (Base, Candidate, CGSR).
# --------------------------------------------------------------------------
EXPECTED_MIOU = {
    "voc": (0.8536171478072758, 0.8589691435052641, 0.8577803683964914),
    "context59": (0.3599881136274712, 0.30870109844639604, 0.36790070114319684),
    "ade20k": (0.16432640866588497, 0.14891131688957823, 0.16462632184272766),
    "coco_stuff27": (0.23884218192073192, 0.2396795170515858, 0.24672267184700136),
    "cityscapes": (0.26456859013662504, 0.2490733769613716, 0.2773282291242289),
}

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260923
