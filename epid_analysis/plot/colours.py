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



def get_smoking_status_colours():
    return {"Never smoker": "green", "Former smoker": "orange", "Current smoker": "red"}
