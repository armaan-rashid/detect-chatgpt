"""
This file contains some basic data processing utility functions. 
Can be run as a script to either repair unfinished data, merge data
or load data from files into the main ChatGPT script. 
"""

import pandas as pd
import data_querying
from argparse import ArgumentParser
from revChatGPT.V1 import Chatbot

def repair_dataframe(data: pd.DataFrame, chatbot: Chatbot, verbose=False):
    """
    DESC: Repair dataframe that has incomplete responses from ChatGPT.
    PARAMS:
    data: a dataFrame that has both a 'prompts' and 'responses' column
    chatbot: logged in ChatGPT
    verbose: print chatGPT's responses while querying 
    """
    fail = 0
    count = 0
    for _, row in data.iterrows():
        if row['responses'] == data_querying.FAILSTRING:
            try: 
                prompt = row['prompts']
                response = data_querying.prompt_ChatGPT(prompt, chatbot)
                row['responses'] = response
                if verbose:
                    print(f'{prompt}:{response}')
                count += 1
                chatbot.reset_chat()
                chatbot.clear_conversations()
            except:
                print(f'The prompt: {prompt} did not successfully get a response from ChatGPT.\n')
                fail += 1
                continue
    print(f'Successfully got {count} responses from ChatGPT, failed to get {fail} responses.')
    return data




def merge_human_sampled(original_file, original_cols, sampled_file, sampled_cols, outfile=None):
    """
    DESC: Given files of both original and sampled data,
    merge them into one dataFrame.
    PARAMS: 
    original_file, sampled_file: file of human data, chatGPT data resp.
    original_cols, sampled_cols: list of cols to read in from original_file, sampled_file resp. 
        if there are multiple columns, they're concatenated with a space separating the strings in each.
    outfile: where to write merged data
    RETURNS: dataFrame of merged data
    """
    original = pd.read_csv(original_file)
    sampled = pd.read_csv(sampled_file)
    
    if original_cols is None:
        original_cols = original.columns
    if sampled_cols is None:
        sampled_cols = sampled.columns

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



def load_data(filenames):
    """
    Load data from files into dict format.
    Cols to be loaded in can be specified 
    For compatibility with existing DetectGPT code.
    Expects that the dfs loaded in has 'original, sampled'
    columns and ignores other columns.
    """
    dfs = [pd.read_csv(filename) for filename in filenames]
    for df in dfs:
        assert 'original' in df.columns and 'sampled' in df.columns, 'files need to have original, sampled cols'
    return [{'original': df['original'].values.tolist(),
             'sampled': df['sampled'].values.tolist()} for df in dfs]


if __name__=='__main__':
    parser = ArgumentParser(prog='process data already retrieved, in different ways')
    parser.add_argument('task', help='what you want to do', choices=['merge', 'load', 'repair'])
    merge = parser.add_argument_group()
    merge.add_argument('--orig_file', help='file with human data')
    merge.add_argument('--orig_cols', help='cols to grab from orig_file')
    merge.add_argument('--sampled_file', help='file with ChatGPT data')
    merge.add_argument('--sampled_cols', help='cols to grab from data')
    merge.add_argument('--outfile', help='where to store new merged data')
    load = parser.add_argument_group()
    load.add_argument('--load_files', nargs='*', help='files to load in from')
    repair = parser.add_argument_group()
    repair.add_argument('--repair_file', nargs=1, help='file with data that needs to be repaired')
    repair.add_argument('--email', nargs=1, help='for ChatGPT login')
    repair.add_argument('--password', nargs=1),
    repair.add_argument('--paid', action='store_true', help='specify if acct holder has paid ChatGPT')

    parser.add_argument('-v', '--verbose', action='store_true', help='print while doing stuff')
    args = parser.parse_args()

    if args.task == 'merge':
        assert args.orig_file and args.sampled_file, 'need to have files to merge!'
        merged = merge_human_sampled(args.orig_file, args.orig_cols, args.sampled_file, args.sampled_cols, args.outfile)
    
    elif args.task == 'load':
        assert (files := args.load_files), 'need to have at least one file to load'
        load_data(files)


    if args.dataset == 'repair':
        broken = pd.read_csv(args.repair_file)
        fixed = repair_dataframe(broken, data_querying.init_ChatGPT(args.email, args.password, args.paid))
        fixed.to_csv(args.repair_file, index=False)

