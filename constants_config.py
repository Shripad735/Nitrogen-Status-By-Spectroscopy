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
DATA_FOLDER_PATH = '../datasets'
