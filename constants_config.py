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
TARGET_VARIABLES_WITH_MEAN = TARGET_VARIABLES + ['mean']
DATA_FOLDER_PATH = '../datasets'

COLOR_PALETTE = {
    'N_Value': ('#ADD8E6', '#E0FFFF'),  # light blue, very light blue
    'SC_Value': ('#90EE90', '#D3FFD3'),  # light green, very light green
    'ST_Value': ('#FFA07A', '#FFDAB9'),  # light orange, very light orange
    'mean': ('#EE82EE', '#DDA0DD')  # light purple, very light purple
}
