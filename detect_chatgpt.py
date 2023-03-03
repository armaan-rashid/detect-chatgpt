"""
Project model.
"""

from argparse import ArgumentParser
import transformers
import functools
from torch import cuda, manual_seed
import torch
import numpy as np
from datetime import time
import tqdm
import query_probabilities as qp
import evaluation 
import pandas as pd
from perturb import generate_perturbations
from matplotlib import pyplot as plt

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MASK_FILLING_MODEL = "t5-3b"    # use for all experiments

manual_seed(0)
np.random.seed(0)

def load_data(filename):
    """
    Load data from file into dict format.
    Expects that the dfs loaded in has 'original, sampled'
    columns and ignores other columns.
    """
    df = pd.read_csv(filename)
    assert 'original' in df.columns and 'sampled' in df.columns, 'files need to have original and sampled cols'
    print(f'Loading data from {filename}.')
    return {'original': df['original'].values.tolist(),
            'sampled': df['sampled'].values.tolist()}


def load_mask_model(mask_model_name):
    """
    DESC: loads and returns mask model and tokenizer
    CALLED BY: perturb_texts
    """
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_model_name) # can be cached. 
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_model_name, model_max_length=n_positions)
    
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()
    mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

    return mask_model, mask_tokenizer

def load_huggingface_model_and_tokenizer(model: str, dataset: str):
    """
    TODO: make this work for multiple models!
    DESC: Load and return a huggingface model with model name.
    """
    print(f'Loading HF model {model}...')
    base_model_kwargs = {}
    if 'gpt-j' in model or 'neox' in model:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model, **base_model_kwargs)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(model, **optional_tok_kwargs)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer




def perturb_texts(data, mask_model, mask_tokenizer, 
                  pct_words_masked=0.3, span_length=2, n_perturbations=1, n_perturbation_rounds=1):
    """ 
    DESC: This function takes in the data and perturbs it according to options passed in.
    PARAMS:
    data: dictionary of human and chatGPT samples. Must have 'original' and 'sampled' keys as such, 
        with lists of strings stored at each location.
    pct_words_masked, span_length: percentage of text to perturb and length of spans to perturb resp.
    n_perturbations: number of perturbed texts to generate for each example
    n_perturbation_rounds: number of times to try perturbing
    RETURNS: 
    perturbed, a List[dict] for every original vs. sampled pair. Each dict is of the form:
    {
        'original': the orig. human passage
        'sampled': ChatGPT passage
        'perturbed_original': a List of perturbed passages of the original
        'perturbed_sampled': a List of perturbed ChatGPT passages
    }
    """
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(generate_perturbations, span_length=span_length, pct=pct_words_masked,
                                   mask_model=mask_model, mask_tokenizer=mask_tokenizer)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    # START unchecked function
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    perturbed = [{"original": original_text[idx], "sampled": sampled_text[idx],
                "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
                "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]} 
                for idx in range(len(original_text))]
    print(f'Created {n_perturbations * 2 * len(perturbed)} texts.')
    return perturbed


def query_lls(results, openai_model=None, openai_opts=None, base_tokenizer=None, base_model=None):
    """
    TODO: make this function work for multiple query models.
    DESC: Given passages and their perturbed versions, query log likelihoods for all of them
    from the query models.
    PARAMS:
    results: a List[Dict] where each dict has original passage, sample passage, and perturbed versions of each
    openai_model: name of openai model as str
    base_tokenizer, base_model: if an HF model used for querying, the actual model and tokenizer
    RETURNS:
    results, but with additional keys in each dict as follows:
    {
        'original_ll', 'sampled_ll': lls of original, sampled passage under query models
        'all_perturbed_sampled_ll','all_perturbed_original_ll': all lls of all perturbed passages
        'perturbed_sampled_ll', 'perturbed_original_ll': average lls over all perturbations
        'perturbed_sampled_ll_std','perturbed_original_ll_std': std. dev of ll over all perturbations, for sampled/orig.
    }
    """

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = qp.get_lls(res["perturbed_sampled"], openai_model, base_tokenizer, base_model, **open_ai_opts)
        p_original_ll = qp.get_lls(res["perturbed_original"], openai_model, base_tokenizer, base_model, **open_ai_opts)
        res["original_ll"] = qp.get_ll(res["original"], openai_model, base_tokenizer, base_model, **open_ai_opts)
        res["sampled_ll"] = qp.get_ll(res["sampled"], openai_model, base_tokenizer, base_model, **open_ai_opts)
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    tokens_used = qp.count_tokens()
    print(f'This query used {tokens_used} tokens.')

    return results


def run_perturbation_experiment(results, criterion, hyperparameters, dataset):
    """
    DESC: Given results of perturbations + probabilities, make probabilistic classification predictions for
    each candidate passage and then evaluate them!

    PARAMS:
    results: List[Dict], where each dict contains an original passage, a ChatGPT passage,
    all their perturbations, and the log probabilities for all these passages. See docstrings
    of query_lls and perturb_texts for more info on what keys are in each dict.
    criterion: 'd' or 'z'. If the criterion is 'd' make a probabilistic pred. between 0 or 1 based on \
        the difference in log likelihoods between a passage and its perturbations. If it's 'z', use \
        the difference divided by the standard dev. of the lls over all perturbations: a z-score. 
    hyperparameters: dict of span_length, pct_words_masked, and n_perturbations
    RETURNS:
    Dict with info and results about experiment!
    """
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            print(f'Making predictions for difference criteria.')
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            print(f'Making predictions for z-score criteria.')
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = evaluation.get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = evaluation.get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'{dataset}_{"difference" if criterion == "d" else "z-score"}_{hyperparameters["n_perturbations"]}_{hyperparameters["perturb_pct"]}.'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': hyperparameters,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }

if __name__ == '__main__':
    parser = ArgumentParser(prog='run detectChatGPT')
    parser.add_argument('dataset', help='name of dataset')
    parser.add_argument('infile', help='csv file: where to read data from')
    parser.add_argument('query_model', help='model to be used for probability querying')
    parser.add_argument('-o', '--openai', action='store_true', help='specify if query model is an OpenAI model')
    parser.add_argument('-d', '--directory', help='directory to save plots')

    perturb_options = parser.add_argument_group()
    perturb_options.add_argument('-n', '--n_perturbations', help='number of perturbations to perform in experiments', type=int, default=25)
    perturb_options.add_argument('-s', '--span_length', help='span of tokens to mask in candidate passages', type=int, default=2)
    perturb_options.add_argument('-p', '--perturb_pct', help='percentage (as decimal) of each passage to perturb', type=float, default=0.15)
    perturb_options.add_argument('-r', '--n_perturbation_rounds', help='number of times to attempt perturbations', type=int, default=1)
    
    open_ai_opts = parser.add_argument_group()
    open_ai_opts.add_argument('-l', '--logprobs', help='how many tokens to include logprobs for', choices=[0,1,2,3,4,5], default=0)
    open_ai_opts.add_argument('-e', '--echo', help='echo both prompt and completion', action='store_true')
    open_ai_opts.add_argument('-m', '--max_tokens', help='max_tokens to be produced in a response', type=int, default=0)
    open_ai_opts.add_argument('-t', '--temperature', help='randomness to use in generation', type=float, default=0.0)
    open_ai_opts.add_argument('-c', '--completions', help='num completions to gen for each prompt', type=int, default=1)

    args = parser.parse_args()

    data = load_data(args.infile)

    hyperparameters = {
        'n_perturbations': args.n_perturbations,
        'span_length': args.span_length,
        'perturb_pct': args.perturb_pct,
        'n_perturbation_rounds': args.n_perturbation_rounds,
    }

    open_ai_hyperparams = {
        'logprobs': args.logprobs,
        'echo': args.echo,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'n': args.completions,
    }

    # core model pipeline: perturb, query probabilities, make predictions
    mask_tokenizer, mask_model = load_mask_model(MASK_FILLING_MODEL)
    perturbed = perturb_texts(data, mask_tokenizer, mask_model, **hyperparameters)
    if args.openai:
        results = query_lls(perturbed, openai_model=args.query_model, openai_opts=open_ai_hyperparams)
    else: 
        hf_model, hf_tokenizer = load_huggingface_model_and_tokenizer(args.query_model, args.dataset)
        results = query_lls(perturbed, base_model=hf_model, base_tokenizer=hf_tokenizer)
    experiments = [run_perturbation_experiment(results, criterion, hyperparameters) for criterion in ['z', 'd']]

    # graph results
    plt.figure()
    evaluation.save_roc_curves(experiments, args.query_model, args.directory)
    evaluation.save_ll_histograms(experiments, args.directory)
    evaluation.save_llr_histograms(experiments, args.directory)
