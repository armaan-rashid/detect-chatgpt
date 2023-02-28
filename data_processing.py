import pandas as pd
import data_querying


def load_data(filename):
    """
    Load data from file into dict format.
    For compatibility with existing DetectGPT code.
    Expects that the df loaded in has 'original, sampled'
    columns.
    """
    df = pd.read_csv(filename)
    return {'original': df['original'].values.tolist(),
            'sampled': df['sampled'].values.tolist()}
