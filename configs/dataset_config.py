DATASET_NAMES = ["Kermany", "PRF", "NIH-RSNA", "CheXpert", "PadChest"]

DATASETS = {
    "Kermany":  "/kaggle/input/datasets/imran50/kermany-pcxr/chest_xray",
    "PRF":      "/kaggle/input/datasets/imran50/sylhet-train-test/sylhet/chest_xray",
    "NIH-RSNA": "/kaggle/input/datasets/imran50/nih-rsna-acxr",
    "CheXpert": "/kaggle/input/datasets/imran50/chexpert",
    "PadChest": "/kaggle/input/datasets/imran50/padchest-8bit",
}

NAME_MAP = {
    "Sylhet": "PRF",
    "NIH": "NIH-RSNA",
    "Padchest": "PadChest"
}

def normalize_dataset_name(name: str) -> str:
    cleaned = str(name).replace("(in-domain)", "").strip()
    return NAME_MAP.get(cleaned, cleaned)

CLASS_WEIGHT_RULES = {
    "Kermany":  {"normal_multiplier": 3.0, "pneumonia_multiplier": 1.0},
    "PRF":      {"normal_multiplier": 1.0, "pneumonia_multiplier": 2.0},
    "NIH-RSNA": {"normal_multiplier": 1.0, "pneumonia_multiplier": 1.0},
    "CheXpert": {"normal_multiplier": 1.0, "pneumonia_multiplier": 1.0},
    "PadChest": {"normal_multiplier": 1.0, "pneumonia_multiplier": 1.0},
}
