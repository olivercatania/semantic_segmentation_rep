ORGANS = (
    "abdominal_wall",
    "colon",
    "inferior_mesenteric_artery",
    "intestinal_veins",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach",
    "ureter",
    "vesicular_glands",
)

IMAGE_SIZE = (512, 640)  # Height, width
NUM_CLASSES = 2
SEED = 42

SPLIT_IDS = {
    "train": {
        "01", "04", "05", "06", "08", "09", "10",
        "12", "15", "16", "17", "19", "22", "23",
        "24", "25", "27", "28", "29", "30", "31",
    },

    "validation": {"03", "21", "26"},

    "test": {
        "02", "07", "11", "13",
        "14", "18", "20", "32",
    },
}