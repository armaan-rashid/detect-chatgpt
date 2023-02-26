from google.cloud import bigquery
import pandas as pd
import transformers
import random
from datasets import load_dataset
from torch import cuda
from custom_datasets import process_spaces
from argparse import ArgumentParser
from revChatGPT.V1 import Chatbot

DEVICE = 'cuda' if cuda.is_available() else 'cpu'

def init_ChatGPT(email, password, paid=False):
    chatbot = Chatbot(config={
        'email': email,
        'password': password,
        'paid': paid
    })
    return chatbot


def prompt_ChatGPT(prompt: str, chatbot: Chatbot):
    response = ''
    for data in chatbot.ask(prompt):
        response = data['message']
    return response


def prompt_from_dataframe(data: pd.DataFrame, chatbot, verbose=False, outfile=None, retain_prompt=False):
    """
    Given a dataframe with a 'prompts' column, query ChatGPT 
    to generate a response for every prompt and
    append these responses to the dataFrame.
    Save the DF as a csv with prompts if retain_prompt
    is True, and print ChatGPT's responses while
    querying if verbose is True.
    """
    count = 0
    responses = []
    for prompt in data['prompts']:
        try: 
            response = prompt_ChatGPT(prompt, chatbot)
            responses.append(response)
            if verbose:
                print(f'response')
            count += 1
        except:
            print(f'The prompt: {prompt} did not successfully get a response from ChatGPT.\n')
            responses.append('Failed response.')
            continue
    data['responses'] = responses   # add responses to the DF
    if outfile:
        if retain_prompt:
            data.to_csv(outfile, index=False)
        else:
            data['responses'].to_csv(outfile, index=False)
    return data 
        

def bigquery_load(sql, outfile):
    """
    Pass a SQL query to bigQuery, 
    save results as JSON in outfile.
    """
    client = bigquery.Client()
    df = client.query(sql).to_dataframe()
    df.to_json(outfile)
    print(f"Received {len(df)} examples from BigQuery.")


def xsum_generate(xsum: pd.DataFrame, tokens=30, prompt_msg=None):
    """
    With an XSum dataset as DataFrame and,
    truncate the news articles in the file and prompt
    ChatGPT with prompt_count words, and a prompt_msg if desired.

    Caller can add a prompt_msg to the beginning of each prompt if 
    caller feels ChatGPT will generate a more appropriate response that way.
    """
    tokenizer = transformers.PreTrainedTokenizer('gpt2-xl')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized = tokenizer(xsum['articles'], return_tensors="pt", padding=True).to(DEVICE)
    tokenized = {key: value[:, :tokens] for key, value in tokenized.items()}

    prompts = tokenizer.batch_decode(tokenized['input_ids'], skip_special_tokens=True)
    xsum['prompts'] = prompts
    if prompt_msg:
        xsum['prompts'] = [prompt_msg + prompt for prompt in prompts]
    xsum = prompt_from_dataframe(xsum, init_ChatGPT(args.email, args.password, args.plus), args.destination, args.retain)
    return xsum

def xsum_load(infile=None, num_examples=500, preprocess=process_spaces):
    """
    Download XSum from HuggingFace datasets hub, or from file.
    Sample num_examples from it.
    Return dataframe, save to outfile if specified.
    """
    if infile:
        return pd.read_csv(infile)
    xsum_dict = load_dataset('xsum')
    xsum = xsum_dict['train']
    articles = [preprocess(xsum[idx]['document']) for idx in random.sample(range(len(xsum)), num_examples)]
    return pd.DataFrame(articles)



    

if __name__ == '__main__':
    argparser = ArgumentParser(prog='ChatGPT Scraper', description='Generate tokens and responses from ChatGPT using unofficial API.')
    argparser.add_argument('email', help='openAI login')
    argparser.add_argument('password', help='openAI login')
    argparser.add_argument('dataset', help="Specify which dataset you want to generate ChatGPT examples for.", choices=['xsum', 'pubmed', 'squad'])
    loader = argparser.add_mutually_exclusive_group(required=True)
    loader.add_argument('-l', '--load', help='if you need to also download your dataset from the Hub, specify this option')
    loader.add_argument('-f', '--file', help='csv file where dataset needs to be loaded from!')
    argparser.add_argument('--plus', action='store_true', help='specify option if you have ChatGPT Plus', default=False)
    argparser.add_argument('-m', '--msg', help='prompt before \'actual\' dataset prompt to give ChatGPT, if that may \
                                                help ChatGPT give a better response')
    argparser.add_argument('-r', '--retain', action='store_true', help='If this option is specified, write both prompt \
                                                                        and response together, separated by a space, in file/output.')
    argparser.add_argument('-d', '--destination', action='store', nargs='?', help='Destination file to write prompts/responses. \
                                                                                   Ignored if files are given.')
    argparser.add_argument('-t', '--tokens', help='Specify number of tokens for creating prompts, when this is ambiguous for a dataset (like xsum ), otherwise ignored',
                            default=30)
    argparser.add_argument('-n', '--num_examples', help='Number of examples to grab when downloading a dataset, ignored otherwise.', default=500)

    args = argparser.parse_args()

    if args.dataset == 'xsum':
        if args.load:
            xsum = xsum_load(num_examples=args.num_examples)
        else:
            xsum = xsum_load(infile=args.file)
        xsum_with_responses = xsum_generate(xsum, args.tokens, args.msg)

