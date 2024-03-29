"""
These are the essential detection functions for detectChatGPT, 
adapted from Eric Mitchell et al.'s original implementation. 
"""

import os
import openai
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
import transformers
import pandas as pd
import time

API_TOKEN_COUNTER = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2')

def write_lls(results, model: str, dataset: str, temp: str): 
    """
    DESC: write log-likelihoods to a dataFrame and an outfile. To avoid money-wastage!
    PARAMS:
    results: List[dict]. Each dict has many keys; the relevant keys are: 
    {
        'original_ll', 'sampled_ll': lls of original, sampled passage under query models
        'all_perturbed_sampled_ll','all_perturbed_original_ll': all lls of all perturbed passages
    }
    model: name of model that generated these results
    """
    model = model.strip('/')    # avoid accidental directories

    original_lls = [[res['original_ll']] + res['all_perturbed_original_ll'] for res in results]
    sampled_lls = [[res['sampled_ll']] + res['all_perturbed_sampled_ll'] for res in results]
    orig_dict = { f'o{i+1}' : ll for i, ll in enumerate(original_lls)}
    sampled_dict = { f's{i+1}' : ll for i, ll in enumerate(sampled_lls)}
    df = pd.DataFrame(data={**orig_dict, **sampled_dict})
    save_dir = f'lls/{dataset}_t{temp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(f'{save_dir}/{model}.csv', index=False)
    return df

def read_lls(filename, n=0, k=0, adversary=False):
    """
    DESC: read in lls from a file formatted by write_lls
    PARAMS: 
    n, k indicate how many perturbations and how many candidates to read in, respectively
    if adversary is true, choose a random perturbation to treat as a candidate passage
    """
    lls = pd.read_csv(filename)
    c = len(lls.columns) // 2  # number of candidate passages
    k = min(c, k) if k != 0 else c
    n = min(n, len(lls) - 1) if n != 0 else len(lls) - 1
    if adversary:   # pick one of the perturbations to use as the 'candidate' sampled passage
        choice = np.random.choice(n)
        results = [{
            'original_ll': lls[f'o{i}'][0],
            'sampled_ll': lls[f's{i}'][choice + 1],
            'all_perturbed_original_ll': lls[f'o{i}'][1:n+1].values.tolist(),
            "perturbed_original_ll": np.mean(lls[f'o{i}'][1:n+1]),
            "perturbed_original_ll_std": np.std(lls[f'o{i}'][1:n+1]) if n > 1 else 1,
            'all_perturbed_sampled_ll': lls[f's{i}'][:choice + 1].values.tolist() + lls[f's{i}'][min(len(lls[f's{i}']), choice + 2):].values.tolist() 
        } for i in range(1,k+1)]
        for i in range(1, k+1):
            results[i-1].update({
                "perturbed_sampled_ll": np.mean(results[i-1]['all_perturbed_sampled_ll']),
                "perturbed_sampled_ll_std": np.std(results[i-1]['all_perturbed_sampled_ll']) if n > 1 else 1,
            })
        return results
    return [{'original_ll': lls[f'o{i}'][0],
            'sampled_ll': lls[f's{i}'][0],
            'all_perturbed_original_ll': lls[f'o{i}'][1:n+1].values.tolist(),
            'all_perturbed_sampled_ll': lls[f's{i}'][1:n+1].values.tolist(),
            "perturbed_sampled_ll": np.mean(lls[f's{i}'][1:n+1]),
            "perturbed_original_ll": np.mean(lls[f'o{i}'][1:n+1]),
            "perturbed_sampled_ll_std": np.std(lls[f's{i}'][1:n+1]) if n > 1 else 1,
            "perturbed_original_ll_std": np.std(lls[f'o{i}'][1:n+1]) if n > 1 else 1} for i in range(1,k+1)]

def get_ll(text, openai_model=None, base_tokenizer=None, base_model=None, **open_ai_opts):
    """
    DESC: The essence of detection: given a candidate text, puts the text
    through a model and has the model assess what its own probability of
    outputting that text (i.e. its loss) is. 
    PARAMS:
    text: string of text to feed to model
    openai_model: self-explanatory
    base_tokenizer, base_model: HuggingFace tokenizer and model to use on text
    RETURNS: float of avg log likelihood of passage
    """
    if openai_model:        
        global API_TOKEN_COUNTER
        openai.api_key = os.getenv('OPENAI_API_KEY')
        while True:
            try:
                r = openai.Completion.create(model=openai_model, prompt=f"<|endoftext|>{text}", **open_ai_opts)
                API_TOKEN_COUNTER += r['usage']['total_tokens']
                result = r['choices'][0]
                tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]
                break
            except:
                print('Hit the rate limit. Just wait!')
                time.sleep(2)


        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        assert base_tokenizer is not None and base_model is not None
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, openai_model=None, base_tokenizer=None, base_model=None, batch_size=50, **open_ai_opts):
    """
    DESC: A wrapper for get_ll, that increments the OPENAI_API_TOKEN counter
    before getting the log-likelihood under the query models, if 
    there's an OpenAI model among the query models.
    PARAMS:
    texts: List[str] to be fed into get_ll
    openai_model, base_tokenizer, base_model: models for querying
    batch_size: if openai_model passed in, number of threads to use for querying at a time
    RETURNS: a List of probabilities
    """

    if not openai_model:
        assert base_tokenizer and base_model, 'need tokenizer and model for ll calculation'
        return [get_ll(text, base_tokenizer=base_tokenizer, base_model=base_model) for text in texts]
    else:
        pool = ThreadPool(batch_size)   # speed things up!
        return pool.map(lambda txt: get_ll(txt, openai_model, **open_ai_opts), texts)

def count_tokens(reset=False):
    if reset:
        global API_TOKEN_COUNTER
        API_TOKEN_COUNTER = 0
    return API_TOKEN_COUNTER   # so other files can access the counter