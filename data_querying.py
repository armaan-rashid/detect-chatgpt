"""
This file implements the functionality for generating ChatGPT passages.

Each dataset has a LOAD function which loads the human dataset from wherever 
it may be: HuggingFace, local files, etc.

Each dataset also has a GENERATE function which takes in a human dataset
and prompts ChatGPT to generate examples in whatever way is appropriate 
to that dataset: asking a question, asking it to complete text, etc.

When run as a script, main() calls a LOAD function to create prompts
and then a GENERATE function to create responses for a dataset. The GENERATE funcs
call the core ChatGPT interfaces prompt_from_dataframe/prompt_ChatGPT. 

There are lots of options for I/O at multiple stages in the querying process.
Generally, we use .csv files and DataFrames because it's easy. 
"""


# from google.cloud import bigquery
import pandas as pd
import transformers
import random
from datasets import load_dataset
from torch import cuda
from data_processing import process_spaces   
from argparse import ArgumentParser
from revChatGPT.V1 import Chatbot

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
FAILSTRING = 'Failed response.'

def init_ChatGPT(login):
    """
    DESC: Login to chatGPT.
    CALLED BY: generate() funcs
    """
    try:
        chatbot = Chatbot(config=login)
        return chatbot
    except:
        print('Can\'t log in right now. Maybe check your internet connection.')


def prompt_ChatGPT(prompt: str, chatbot: Chatbot, min_words=250, cont_prompt='Please, keep going.'):
    """
    DESC: Self-explanatory, prompts chatbot with prompt
    til response length greater than min_words.
    CALLED_BY: generate() funcs
    """
    response = ''
    for data in chatbot.ask(prompt):
        response = data['message']
    while len(response) < min_words:
        append = ''
        for data in chatbot.ask(cont_prompt):
            append = data['message']
        response = response + ' ' + append
    return response

def output_ChatGPT(df: pd.DataFrame, outfile, retain: bool):
    """
    DESC: print a dataFrame to csv that contains prompts, responses
    CALLED BY: all generate() funcs
    PARAMS: 
    df: DataFrame to print
    retain: indicating to print prompts or not
    outfile: file to print to
    """
    if retain:
        df['prompts', 'responses'].to_csv(outfile, index=False)
    else:
        df['responses'].to_csv(outfile, index=False)


def prompt_from_dataframe(data: pd.DataFrame, chatbot: Chatbot, verbose=False, min_words=250):
    """
    DESC: Query ChatGPT to generate a response for every prompt and
    append these responses to a dataFrame.
    PARAMS:
    data: dataFrame with prompts in it
    chatbot: ChatGPT already logged in
    verbose: print ChatGPT's responses or not
    min_words: min length of valid response from ChatGPT
    RETURNS:
    df: dataFrame with prompts and responses
    """
    count = 0
    fail = 0
    responses = []
    for prompt in data['prompts']:
        try: 
            response = prompt_ChatGPT(prompt, chatbot, min_words)
            responses.append(response)
            if verbose:
                print(f'{prompt}:{response}')
            count += 1
            chatbot.reset_chat()
        except:
            print(f'The prompt: {prompt} did not successfully get a response from ChatGPT. It is likely \
                    you have hit the request limit, so we\'ll stop right now. \n')
            fail = len(data) - count
            responses.extend([FAILSTRING for _ in range(fail)])
            break
    data['responses'] = responses   # add responses to the DF
    print(f'Successfully got {count} responses from ChatGPT, failed to get {fail} responses.')
    return data 
        

# def bigquery_load(sql, outfile):
#     """
#     Pass a SQL query to bigQuery, 
#     save results as JSON in outfile.
#     """
#     client = bigquery.Client()
#     df = client.query(sql).to_dataframe()
#     df.to_json(outfile)
#     print(f"Received {len(df)} examples from BigQuery.")


def xsum_generate(xsum: pd.DataFrame, tokens=30, prompt_msg=None, min_words=250, retain=False, outfile=None):
    """
    DESC: Truncate the news articles in the XSum data and prompt
    ChatGPT. This function is different than the functions for other datasets
    because we're calling a tokenizer to cut off the prompt at 
    the length specified by tokens, whereas the other datasets have a natural 
    notion of prompt. Part of this function adapted from Mitchell et al.'s
    original ChatGPT implementation @ https://github.com/eric-mitchell/detect-gpt
    PARAMS: 
    xsum: DataFrame of XSum news articles (needs 'articles' column)
    tokens: number of tokens from article to prompt ChatGPT with
    prompt_msg: add'l message to prompt ChatGPT with BEFORE news article
    min_words: min length of valid response from ChatGPT
    retain: write prompts to outfile if True
    outfile: file to write prompts/responses to
    RETURNS: DataFrame of generated XSum examples
    """
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized = tokenizer(xsum['articles'].values.tolist(), return_tensors="pt", padding=True).to(DEVICE)
    tokenized = {key: value[:, :tokens] for key, value in tokenized.items()}

    prompts = tokenizer.batch_decode(tokenized['input_ids'], skip_special_tokens=True)
    xsum['prompts'] = prompts
    
    if prompt_msg:
        xsum['prompts'] = [prompt_msg + prompt for prompt in prompts]
    xsum = prompt_from_dataframe(xsum, init_ChatGPT(LOGIN), verbose=VERBOSE, min_words=min_words)
    if outfile:
        if retain:
            xsum['prompts', 'responses'].to_csv(outfile, index=False)
        else:
            xsum['responses'].to_csv(outfile, index=False)
    return xsum

def xsum_load(infile=None, outfile=None, num_examples=500, preprocess=process_spaces):
    """
    DESC: Download XSum from HuggingFace datasets hub, or load from file.
    PARAMS:
    infile: file where dataset already lives, if applicable
    outfile: file to write human data to if applicable
    num_examples: num to take from HuggingFace dataset
    preprocess: function for preprocessing examples
    RETURNS: DataFrame of human XSum examples
    """
    if infile:
        return pd.read_csv(infile)
    xsum_dict = load_dataset('xsum')
    xsum = xsum_dict['train']
    articles = [preprocess(xsum[idx]['document']) for idx in random.sample(range(len(xsum)), num_examples)]
    df = pd.DataFrame(articles)
    if outfile:
        df.to_csv(outfile, index=False)
    return df


def squad_generate(squad: pd.DataFrame, min_words=250, retain=False, outfile=None):
    """
    DESC: Given a dataFrame of SQuAD q's, a's, contexts, prepare data
    to feed in as prompts to ChatGPT. Write to outfile if provided.
    PARAMS:
    squad: DataFrame of squad examples (must have contexts and questions cols)
    prompt_msg: msg to prompt chatGPT with in addition to questions
    min_words: min valid length of chatGPT response
    retain: write prompts to outfile
    outfile: file to write prompts/responses
    RETURNS:
    squad: DataFrame with chatGPT responses
    """
    squad['prompts'] = squad.apply(lambda row: row['contexts'] + '\n' + row['questions'], axis=1)
    squad = prompt_from_dataframe(squad, init_ChatGPT(LOGIN), min_words=min_words, verbose=VERBOSE)
    if outfile:
        if retain:
            squad['prompts', 'responses'].to_csv(outfile, index=False)
        else:
            squad['responses'].to_csv(outfile, index=False)
    return squad


def squad_load(infile=None, outfile=None, num_examples=500, preprocess=process_spaces):
    """
    DESC: Download SQuAD from HuggingFace hub, or from file.
    Sample num_examples if downloading.
    PARAMS:
    infile: file with already loaded data
    outfile: file to write human data
    num_examples: number to sample if downloading from HuggingFace
    preprocess: preprocessor function to apply to each example
    RETURNS:
    dataFrame with contexts, questions, and answers
    """
    if infile:
        return pd.read_csv(infile)
    squad_dict = load_dataset("squad")
    squad = squad_dict['train']
    idxs = random.sample(range(len(squad)), num_examples)
    contexts = [preprocess(squad[idx]['context']) for idx in idxs]
    questions = [preprocess(squad[idx]['question']) for idx in idxs]
    answers = [preprocess(squad[idx]['answers']['text']) for idx in idxs]
    df = pd.DataFrame({'contexts': contexts, 'questions': questions, 'answers': answers})
    if outfile:
        df.to_csv(outfile, index=False)
    return df
    



if __name__ == '__main__':
    argparser = ArgumentParser(prog='ChatGPT Scraper', description='Generate tokens and responses from ChatGPT using unofficial API.')
    argparser.add_argument('email', help='openAI login')
    argparser.add_argument('password', help='openAI login')
    argparser.add_argument('-p', '--paid', action='store_true', help='specify option if you have ChatGPT Plus', default=False)
    argparser.add_argument('dataset', help="Specify which dataset you want to generate ChatGPT examples for.", choices=['xsum', 'wp', 'squad'])

    input = argparser.add_mutually_exclusive_group(required=True)
    input.add_argument('-l', '--load', action='store_true', help='if you need to also download your dataset from the Hub, specify this option')
    input.add_argument('-i', '--infile', nargs='+', help='csv file where dataset needs to be loaded from!')
    
    output = argparser.add_argument_group()
    output.add_argument('--out_human', help='If --load is specified, this is where load will store the human language data.')
    output.add_argument('--out_chatgpt', action='store', help='Destination file to write prompts/responses from ChatGPT.')
    output.add_argument('-r', '--retain', action='store_true', help='If this option is specified, write both prompt \
                                                                     and response together, separated by a space, in file/output.')
    output.add_argument('-v', '--verbose', action='store_true', help='Print ChatGPT\'s responses as it\'s being queried.')

    prompt_opts = argparser.add_argument_group()
    prompt_opts.add_argument('-m', '--msg', help='prompt before \'actual\' dataset prompt to give ChatGPT, if that might help ChatGPT give a better response')
    prompt_opts.add_argument('-t', '--tokens', help='Specify number of tokens for creating prompts, when needed for a dataset', default=30, type=int)
    prompt_opts.add_argument('-n', '--num_examples', help='Number of examples to grab when downloading a dataset, ignored otherwise.', type=int, default=500)
    prompt_opts.add_argument('-w','--min_words', help='min_words desired from a ChatGPT response', type=int, default=250)
    
    args = argparser.parse_args()

    # The only global variables, since we may login to ChatGPT multiple times. 
    LOGIN = { 'email': args.email, 'password': args.password, 'paid': args.paid }
    VERBOSE = args.verbose

    if args.dataset == 'xsum':
        if args.load:
            xsum = xsum_load(outfile=args.out_human, num_examples=args.num_examples)
        else:
            xsum = xsum_load(infile=args.infile)
        xsum_with_responses = xsum_generate(xsum, tokens=args.tokens, prompt_msg=args.msg, min_words=args.min_words, 
                                            retain=args.retain, outfile=args.out_chatgpt)

    if args.dataset == 'squad':
        if args.load:
            squad = squad_load(outfile=args.out_human, num_examples=args.num_examples)
        else:
            squad = squad_load(infile=args.infile)
        squad_with_responses = squad_generate(squad, prompt_msg=args.msg, min_words=args.min_words, retain=args.retain, outfile=args.out_chatgpt)
