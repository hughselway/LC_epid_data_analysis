import sci_palettes  # type: ignore


PALETTES = sci_palettes.palettes.PALETTES


def get_histology_colours():
    palette = PALETTES["nejm"]
    return {
        "lung_squamous_cell_carcinoma": palette["TallPoppy"],
        "LUSC": palette["TallPoppy"],
        "lung_adenocarcinoma": palette["WildBlueYonder"],
        "LUAD": palette["WildBlueYonder"],
        "other": palette["Salomie"],
    }


def get_histology_colour(histology):
    return get_histology_colours()[histology]


def get_dataset_colours():
    palette = PALETTES["lancet_lanonc"]
    return {
        "UK Biobank": palette["BondiBlue"],
        "PLCO": palette["MonaLisa"],
        "Combined": palette["TrendyPink"],
    }


def get_dataset_colour(dataset):
    return get_dataset_colours()[dataset]


def get_dataset_cmaps():
    return {
        "UK Biobank": "Blues",
        "PLCO": "Oranges",
        "Combined": "Greens",
    }


def get_dataset_cmap(dataset):
    return get_dataset_cmaps()[dataset]


def get_smoking_status_colours():
    return {"Never smoker": "green", "Former smoker": "orange", "Current smoker": "red"}
