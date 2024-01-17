from pandas import DataFrame, Series

import datetime as dt
import pickle

from typing import *


def parse_date(date: str, f: str="%Y-%m-%d %H:%M:%S")->dt.datetime:
    """Parse date from given format.

    Parameters
    ----------
    date: str
        Date to parse
    f: str
        Format string of date to be parsed

    Returns
    -------
        Date in datetime object
    """
    return dt.datetime.strptime(date, f)

def mask_df(df: DataFrame, mask: Series, invert=False):
    """Simple masking function.

    Parameters
    ----------
    df: DataFrame
        Dataframe to mask
    mask: Series[bool]
        Series of booleans with same length as dataframe index
    invert: bool
        Whether to invert mask
    Returns
    -------
        Masked dataframe
    """
    assert df.index.size == mask.size, f'Mismatched dataframe and mask size, got {df.index.size} and {mask.size}'
    assert mask.dtype == bool, f'Incorrect mask datatype. Should be boolean and got{mask.dtype}'
    return df[mask] if not invert else df[~mask]


def calculate_length_of_stay(f, units='seconds'):
    """Calculate length of stay.

    Parameters
    ----------
    f
    units

    Returns
    -------

    """
    diff = f['VISIT_END_DATETIME'] - f['VISIT_START_DATETIME']
    seconds = diff.total_seconds()
    if units == 'seconds':
        return seconds
    minutes = seconds / 60.
    if units == 'minutes':
        return minutes
    hours = minutes / 60.
    if units == 'hours':
        return minutes
    days = hours / 24.
    if units == 'days':
        return days
    else:
        raise ValueError(f'{units} is not a valid unit for length of stay. Should be in {valid_los}')

def make_numerical(s):
    try:
        s = float(s)
    except:
        s = 0.0
    return s

# Make label processing based on extracted times
def label_processing(data):
    labels = data
    # Initialize masks
    masks = []
    # extract mask based on label times
    for start, stop in zip(labels['start'], labels['stop']):  # TODO: Add start and stop to labels in construction.
        mask = None
        masks.append(mask)
    labels['mask'] = masks
    return labels

def read_graph(graph_path):
    return pickle.load(open(graph_path, 'rb'))
