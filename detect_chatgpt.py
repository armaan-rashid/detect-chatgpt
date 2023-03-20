"""
Project model. Again, some of the funcs adapted from
https://github.com/eric-mitchell/detect-gpt and noted as such.
"""

from argparse import ArgumentParser
import transformers
from torch import cuda, manual_seed
import torch
import numpy as np
import tqdm
import query_probabilities as qp
import evaluation as eval
from perturb import perturb_texts, load_perturbed, write_perturbed
from data_processing import load_data
import os
import copy
import itertools
from glob import glob

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MASK_FILLING_MODEL = "t5-3b"    # use for all experiments

manual_seed(0)
np.random.seed(0)


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
    mask_model.to(DEVICE)
    print('DONE')

    return mask_model, mask_tokenizer

def load_hf_models_and_tokenizers(models: str, dataset: str):
    """
    DESC: Load and return huggingface models with model names.
    """
    base_models = []
    base_tokenizers = []
    for model in models: 
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
        try: base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        except: base_tokenizer.pad_token_id = [base_tokenizer.eos_token_id]
        print(f'MOVING HF MODEL {model} AND TOKENIZER TO GPU if possible.')
        try: base_model.to(DEVICE)
        except: pass
        try: base_tokenizer.to(DEVICE)
        except: pass
        base_models.append(base_model)
        base_tokenizers.append(base_tokenizer)

    return base_models, base_tokenizers


def query_lls(results, openai_models=None, openai_opts=None, base_tokenizers=None, base_models=None):
    """
    DESC: Given passages and their perturbed versions, query log likelihoods for all of them
    from the query models.
    PARAMS:
    results: a List[Dict] where each dict has original passage, sample passage, and perturbed versions of each
    openai_models: list of openai models 
    base_tokenizers, base_models: if an HF model used for querying, a list of the actual model and tokenizer
    RETURNS:
    results, but with additional keys in each dict as follows:
    {
        'original_ll', 'sampled_ll': lls of original, sampled passage under query models
        'all_perturbed_sampled_ll','all_perturbed_original_ll': all lls of all perturbed passages
        'perturbed_sampled_ll', 'perturbed_original_ll': average lls over all perturbations
        'perturbed_sampled_ll_std','perturbed_original_ll_std': std. dev of ll over all perturbations, for sampled/orig.
    }
    """
    all_results = []

    # run for all openai models
    for openai_model in openai_models: 
        results_copy = copy.deepcopy(results)
        for res in tqdm.tqdm(results_copy, desc="Computing log likelihoods"):
            res["original_ll"] = qp.get_ll(res["original"], openai_model, None, None, **openai_opts)
            res["sampled_ll"] = qp.get_ll(res["sampled"], openai_model, None, None, **openai_opts)
            res["all_perturbed_original_ll"] = qp.get_lls(res["perturbed_original"], openai_model, None, None, **openai_opts)
            res["all_perturbed_sampled_ll"] = qp.get_lls(res["perturbed_sampled"], openai_model, None, None, **openai_opts)
            res["perturbed_sampled_ll"] = np.mean(res['all_perturbed_sampled_ll'])
            res["perturbed_original_ll"] = np.mean(res['all_perturbed_original_ll'])
            res["perturbed_sampled_ll_std"] = np.std(res['all_perturbed_sampled_ll']) if len(res['all_perturbed_sampled_ll']) > 1 else 1
            res["perturbed_original_ll_std"] = np.std(res['all_perturbed_original_ll']) if len(res['all_perturbed_original_ll']) > 1 else 1
        tokens_used = qp.count_tokens()
        print(f'This query used {tokens_used} tokens.')
        all_results.append(results_copy)

    # run for all hugging facemodels
    for base_tokenizer, base_model in zip(base_tokenizers, base_models):
        results_copy = copy.deepcopy(results)
        for res in tqdm.tqdm(results_copy, desc="Computing log likelihoods"):
            p_sampled_ll = qp.get_lls(res["perturbed_sampled"], None, base_tokenizer, base_model)
            p_original_ll = qp.get_lls(res["perturbed_original"], None, base_tokenizer, base_model)
            res["original_ll"] = qp.get_ll(res["original"], None, base_tokenizer, base_model)
            res["sampled_ll"] = qp.get_ll(res["sampled"], None, base_tokenizer, base_model)
            res["all_perturbed_sampled_ll"] = p_sampled_ll
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
        all_results.append(results_copy)
    
    return all_results


def run_perturbation_experiment(all_results, criterion, hyperparameters, dataset, temp):
    """
    DESC: Given results of perturbations + probabilities, make probabilistic classification predictions for
    each candidate passage and then evaluate them!

    PARAMS:
    all_results: List[List[Dict]], where each List[Dict] is the result from a single query model, 
        and each dict contains an original passage, a ChatGPT passage,
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
    all_predictions = []
    for query_model_results in all_results: 
        query_model_predictions = {'real': [], 'samples': []}
        for res in query_model_results:
            if criterion == 'd':
                query_model_predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
                query_model_predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
            elif criterion == 'z':
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
                query_model_predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
                query_model_predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])
        all_predictions.append(query_model_predictions)

    # composing model results by taking mean of all discrepancy scores
    final_composed_predictions = {'real': [], 'samples': []}
    for model_prediction in all_predictions: 
        if len(final_composed_predictions['real']) == 0: 
            final_composed_predictions['real'] = model_prediction['real']
        else: 
            final_composed_predictions['real'] = np.add(final_composed_predictions['real'], model_prediction['real'])
        if len(final_composed_predictions['samples']) == 0: 
            final_composed_predictions['samples'] = model_prediction['samples']
        else: 
            final_composed_predictions['samples'] = np.add(final_composed_predictions['samples'], model_prediction['samples'])
    final_composed_predictions['real'] = [x / len(all_predictions) for x in final_composed_predictions['real']]
    final_composed_predictions['samples'] = [x / len(all_predictions) for x in final_composed_predictions['samples']]

    fpr, tpr, roc_auc = eval.get_roc_metrics(final_composed_predictions['real'], final_composed_predictions['samples'])
    p, r, pr_auc = eval.get_precision_recall_metrics(final_composed_predictions['real'], final_composed_predictions['samples'])
    name = f'{dataset}_t{temp}_n{hyperparameters["n_perturbations"]}_{"discrepancy" if criterion == "d" else "z-score"}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': final_composed_predictions,
        'info': hyperparameters,
        'criterion': criterion,
        'raw_results': all_results,
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


def evaluate_and_graph(experiments, dataset, temp, n_perturbed, model_list, adversarial=False):
    """
    DESC: evaluate the experiments and graph and save everything!
    PARAMS: 
    experiments: predictions of perturbation results
    dataset, temp, n_perturbed: dataset used, its temperature and number of perturbations used
    model_list: list of models involved!
    """
    print(f'The results for the models in {model_list} are:')
    save_dir = f'Results/{dataset}_t{temp}_n{n_perturbed}/'
    if adversarial:
        save_dir = f'Results/adversarial/{dataset}_t{temp}_n{n_perturbed}/'
    for model in model_list:
        save_dir += f'{model}_'
    save_dir = save_dir[:-1]    # remove trailing underscore!
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eval.save_roc_curves(experiments, save_dir)
    if len(openai_models + hf_model_names) == 1: 
        eval.save_ll_histograms(experiments[0]["raw_results"][0], save_dir)
        eval.save_llr_histograms(experiments[0]["raw_results"][0], save_dir)
    eval.save_scores(experiments, save_dir)


if __name__ == '__main__':
    parser = ArgumentParser(prog='run detectChatGPT')
    parser.add_argument('dataset', help='name of dataset')
    parser.add_argument('temperature', help='temperature of dataset', choices=['00', '50', '100'])
    parser.add_argument('--openai_query_models', help='openai models to be used for probability querying', nargs="*", default=[])
    parser.add_argument('--huggingface_query_models', help='huggingface models to be used for probability querying', nargs="*", default=[])
    parser.add_argument('-k', '--k_examples', help='load k examples from file', type=int, default=0)
    parser.add_argument('--combine', action='store_true', help='conduct experiments on all combinations of models passed in')
    parser.add_argument('-a', '--adversarial', action='store_true', help='run adversarial experiment. only works if args.ll_files passed in!')


    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--candidate_file', help='csv files: where to read candidate_files from for perturbation')
    inputs.add_argument('--perturbation_file', help='csv files to read perturbations from')
    inputs.add_argument('--ll_files', help='if lls have already been queried, put the filenames here. DO NOT USE THIS IN COMBINATION \
                                            with candidate_file or perturbation_file.', nargs='*')

    perturb_options = parser.add_argument_group()
    perturb_options.add_argument('-n', '--n_perturbations', help='number of perturbations to perform in experiments', type=int, default=100)
    perturb_options.add_argument('-s', '--span_length', help='span of tokens to mask in candidate passages', type=int, default=2)
    perturb_options.add_argument('-p', '--perturb_pct', help='percentage (as decimal) of each passage to perturb', type=float, default=0.15)
    perturb_options.add_argument('-r', '--n_perturbation_rounds', help='number of times to attempt perturbations', type=int, default=1)
    perturb_options.add_argument('-w', '--writefile', help='file to write perturbed examples to')
    
    open_ai_opts = parser.add_argument_group()
    open_ai_opts.add_argument('-l', '--logprobs', help='how many tokens to include logprobs for', choices=[0,1,2,3,4,5], default=0)
    open_ai_opts.add_argument('-e', '--echo', help='echo both prompt and completion', action='store_true')
    open_ai_opts.add_argument('-m', '--max_tokens', help='max_tokens to be produced in a response', type=int, default=0)
    open_ai_opts.add_argument('-c', '--completions', help='num completions to gen for each prompt', type=int, default=1)

    args = parser.parse_args()


    hyperparameters = {
        'n_perturbations': args.n_perturbations,
        'span_length': args.span_length,
        'perturb_pct': args.perturb_pct,
        'n_perturbation_rounds': args.n_perturbation_rounds,
        'temp': args.temperature
    }

    open_ai_hyperparams = {
        'logprobs': args.logprobs,
        'echo': args.echo,
        'max_tokens': args.max_tokens,
        'n': args.completions,
    }

    # core model pipeline: perturb if needed, query probabilities, make predictions

    if args.candidate_file:   # if only candidate passages passed in, generate perturbations!  
        data = load_data(args.candidate_file, args.k_examples)
        mask_tokenizer, mask_model = load_mask_model(MASK_FILLING_MODEL)
        perturbed = perturb_texts(data, mask_tokenizer, mask_model, **hyperparameters)
        if args.writefile:  # write the perturbations if file specified
            write_perturbed(perturbed, args.writefile)

    elif args.perturbation_file:
        perturbed = load_perturbed(args.perturbation_file, args.n_perturbations, args.k_examples)

    all_results = []

    openai_models = []
    if args.openai_query_models:
        assert args.candidate_file or args.perturbation_file, 'you need to have given a file of passages to query probs.'
        openai_models = args.openai_query_models

    hf_model_names, hf_models, hf_tokenizers = [], [], []
    if args.huggingface_query_models:
        assert args.candidate_file or args.perturbation_file, 'you need to have given a file of passages to query probs.'
        hf_model_names = [model[model.rfind('/') + 1:] for model in args.huggingface_query_models]
        hf_models, hf_tokenizers = load_hf_models_and_tokenizers(args.huggingface_query_models, args.dataset)

    if len(openai_models) > 0 or len(hf_models) > 0:
        all_results = query_lls(perturbed, openai_models, openai_opts=open_ai_hyperparams, base_models=hf_models, base_tokenizers=hf_tokenizers)
        for results, model in zip(all_results, openai_models + hf_model_names):
            qp.write_lls(results, model, args.dataset, args.temperature)

    file_models = []
    if args.ll_files:   # probability results that are already done 
        files = glob(args.ll_files) if '*' in args.ll_files else args.ll_files
        file_models = [file[file.rfind('/') + 1:file.find('.csv')] for file in files]
        all_results = [qp.read_lls(ll_file, args.n_perturbations, args.k_examples, args.adversarial) for ll_file in files]
    
    all_models = openai_models + hf_model_names if not args.ll_files else file_models

    if not args.combine:
        experiments = [run_perturbation_experiment(all_results, criterion, hyperparameters, args.dataset, args.temperature) for criterion in ['z', 'd']]
        evaluate_and_graph(experiments, args.dataset, args.temperature, args.n_perturbations, all_models)
    
    if args.combine:
        for choices in range(1, len(all_models) + 1):
            for combo in itertools.combinations(zip(all_models, all_results, strict=True), choices):
                models, results = zip(*combo)
                models, results = list(models), list(results)
                experiments = [run_perturbation_experiment(results, criterion, hyperparameters, args.dataset, args.temperature) for criterion in ['z','d']]
                evaluate_and_graph(experiments, args.dataset, args.temperature, args.n_perturbations, models, args.adversarial)
