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


DATA_FOLDER = 'datasets'
