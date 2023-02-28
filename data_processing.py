import pandas as pd
import data_querying


def merge_human_sampled(original_file, original_cols, sampled_file, sampled_cols, outfile=None):
    """
    Given files of both original and sampled data,
    merge them into one dataFrame according to 
    original_cols and sampled_cols, concatenating 
    between cols as necessary.
    """
    original = pd.read_csv(original_file)
    sampled = pd.read_csv(sampled_file)
    
    def concat_cols(row, cols):
        string = ''
        for col in cols:
            string += row[col] + ' '
        return string.strip()

    original['original'] = original.apply(lambda row: concat_cols(row, original_cols), axis=1)
    sampled['sampled'] = sampled.apply(lambda row: concat_cols(row, sampled_cols), axis=1)
    df = pd.concat([original['original'], sampled['sampled']], axis=1)
    if outfile:
        df.to_csv(outfile, index=False)
    return df



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
