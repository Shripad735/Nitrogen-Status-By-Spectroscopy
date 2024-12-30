from enum import Enum


class ColumnName(str, Enum):
    n_value = 'N_Value'
    sc_value = 'SC_Value'
    st_value = 'ST_Value'
    id = 'ID'


class IDComponents(str, Enum):
    crop = 'crop'
    part = 'part'
    location = 'location'
    date = 'date'


class Crop(str, Enum):
    citrus = 'cit'


class CropPart(str, Enum):
    leaf = 'lea'


TARGET_VARIABLES = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]
NON_FEATURE_COLUMNS = [ColumnName.id.value] + TARGET_VARIABLES
MEAN = 'Mean'
TARGET_VARIABLES_WITH_MEAN = TARGET_VARIABLES + [MEAN]
MULTI = 'Multi'
PLSR_BASED_MODEL = 'PLSR_based_model'
MODELS = [MULTI, PLSR_BASED_MODEL]
TARGET_VARIABLES_WITH_MULTI = TARGET_VARIABLES + [MULTI]
TARGET_VARIABLES_WITH_MULTIS = TARGET_VARIABLES + [f'{MULTI}_{var}' for var in TARGET_VARIABLES]
FIGURE_FOLDER_PATH = 'figures'
DATA_FOLDER_PATH = '../datasets'
DATA_FOLDER = 'datasets'

COLOR_PALETTE_FOR_TARGET_VARIABLES = {
    f'{ColumnName.n_value.value}': ('#ADD8E6', '#E0FFFF'),  # light blue, very light blue
    f'{ColumnName.sc_value.value}': ('#90EE90', '#D3FFD3'),  # light green, very light green
    f'{ColumnName.st_value.value}': ('#FFA07A', '#FFDAB9'),  # light orange, very light orange
    f'{MULTI}': ('#EE82EE', '#DDA0DD'),  # light purple, very light purple
    f'{MEAN}': ('#FFD700', '#FFFACD')  # gold, light goldenrod yellow
}

COLOR_PALETTE_FOR_TWO_MODELS = {
    'model1': '#AEC6CF',  # Light Blue
    'model2': '#FFB347'  # Light Orange
}

