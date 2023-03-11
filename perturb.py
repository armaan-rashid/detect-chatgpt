"""
This file contains the perturbation functionality.
Can be used in other modules or can also be run as
its own script. Largely adapted from Mitchell et al.'s
original implementation @ https://github.com/eric-mitchell/detect-gpt.
"""


import numpy as np
import tqdm
import transformers
import re
import torch
from torch import cuda
import functools
from argparse import ArgumentParser
import pandas as pd


DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MASK_PATTERN = re.compile(r"<extra_id_\d+>")

def load_data(filename, k=None):
    """
    From data_processing, copied here to avoid having to import
    and create package issues on GPUs.
    """
    df = pd.read_csv(filename)
    assert 'original' in df.columns and 'sampled' in df.columns, 'files need to have original and sampled cols'
    print(f'Loading data from {filename}.')
    conv = {'original': df['original'].values.tolist(),
            'sampled': df['sampled'].values.tolist()}
    if k is None:
        k = len(conv['original'])
    conv['original'] = conv['original'][:min(k, len(conv['original']))]
    conv['sampled'] = conv['sampled'][:min(k, len(conv['sampled']))]
    return conv



def load_perturbed(filename, n=0):
    """
    Load perturbed examples from a csv file.
    DataFrame stored in file expected to be in 
    format where for each original, sampled candidate
    passage, the candidate is first in column with all
    perturbations following.
    PARAMS:
    filename: (.csv) where dataFrame is stored
    n: number of perturbations to load
    """
    perturbed = pd.read_csv(filename)
    c = len(perturbed.columns) // 2  # number of candidate passages
    n = min(n, len(perturbed) - 1) if n != 0 else len(perturbed) - 1
    perturbed = [{"original": perturbed[f'o{i}'][0], "sampled": perturbed[f's{i}'][0],
                "perturbed_sampled": perturbed[f's{i}'][1:n+1].values.tolist(),
                "perturbed_original": perturbed[f'o{i}'][1:n+1].values.tolist()} 
                for i in range(1,c+1)]
    return perturbed


def write_perturbed(perturbed, outfile):
    """
    Write perturbations to a file, given a list of dictionaries
    with original text, sampled text, and perturbed versions of each.
    Opposite of load_perturbed.
    """
    original_cols = [[p['original']] + p['perturbed_original'] for p in perturbed]
    sampled_cols = [[p['sampled']] + p['perturbed_sampled'] for p in perturbed]
    orig_dict = { f'o{i+1}' : col for i, col in enumerate(original_cols)}
    sampled_dict = { f's{i+1}' : col for i, col in enumerate(sampled_cols)}
    df = pd.DataFrame(data={**orig_dict, **sampled_dict})
    df.to_csv(outfile, index=False)
    return df

def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
    """
    DESC: masks portions of a given text. 
    RETURNS: the text, with randomly chosen word sequences masked with "<extra_id_i>" instead, where i 
    is the mask number 
    CALLED BY: generate_perturbations_
    PARAMS: 
    span_length: the number of words in a "span", i.e. a single masked sequence
    pct: the percent of text that should be perturbed (percentage of words)
    ceil_pct: round number of masked sequences up if true, down if false. 
    buffer_size: the number of unmasked words that must be between each masked sequence
    """
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    # calculate n_span, which is the number of masked sequences we want. 
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2) # pct * len(tokens) is the number of words to be perturbed. divide by the length of a span to get the number of spans in the text. 
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    # n_masks is the number of masked sequences we got so far. 
    n_masks = 0 
    while n_masks < n_spans:
        # choose start and end index for a possible masked sequence
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]: # if neither the word directly before or after are masks + masks aren't in the selected region...
            tokens[start:end] = [mask_string] # collapse ALL chosen tokens to one token, mask_string, in the array
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    """
    DESC: counts the number of masks in each text in texts.  
    RETURNS: an array of numbers, each of which is the number of masks for the corresponding text in texts
    CALLED BY: replace_masks
    PARAMS: 
    texts: an array of texts. 
    """
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

@torch.no_grad()
def replace_masks(masked_texts, mask_model: transformers.T5ForConditionalGeneration, mask_tokenizer: transformers.T5Tokenizer):
    """
    DESC: return a sample from T5 mask_model for each masked span
    CALLED BY: generate_perturbations_
    PARAMS: 
    texts: an array of already masked texts
    mask_model: masking model (i.e. T5-3B)
    mask_tokenizer: the T5 tokenizer
    RETURNS: texts (List[str]) with the masking tokens and new fills

    """
    n_expected = count_masks(masked_texts) # number of masks in texts
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0] # the integer id corresponding to the masking string's token
    tokens = mask_tokenizer(masked_texts, return_tensors="pt", padding=True).to(DEVICE) # tokenize the texts
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False) # convert tokenized texts (list of integer ids) back to text strings


def extract_fills(texts):
    """
    DESC: return the text without the mask tokens
    CALLED BY: generate_perturbations_
    PARAMS:
    texts: List[str] of texts that have been filled without having the masked strings removed
    RETURNS:
    extracted_fills: just the fills as a list[str] for the maskstrings
    """
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [MASK_PATTERN.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills



def apply_extracted_fills(masked_texts: list, extracted_fills):
    """
    DESC: insert the generated text into the pre-existing text 
    CALLED BY: generate_perturbations_
    PARAMS:
    masked_texts: list of masked texts as strings
    extracted_fills: fill texts to place in the masks in each masked_text
    RETURNS:
    texts: filled with perturbations, no maskstrings
    """
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def generate_perturbations_(texts, span_length, pct, mask_model, mask_tokenizer, ceil_pct=False):
    """
    DESC: Actually perturb texts. Broken into four stages: 
    (1) tokenize text and mask different small spans of it 
    (2) generate reasonable text for each masked span using T5-3b, 
    leaving text with new fills and maskstrings
    (3) extract the new fills
    (4) implant the new fills into the text to REPLACE the masked strings
    PARAMS:
    texts: list of texts to perturb
    span_length: how many tokens at a time to mask during perturbation
    pct: pct of text overall to perturb
    RETURNS the perturbed texts
    """
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts


def generate_perturbations(texts, span_length, pct, mask_model, mask_tokenizer, ceil_pct=False, chunk_size=10):
    """
    DESC: tqdm/progress bar wrapper for generate_perturbations_  
    CALLED BY: get_perturbation_results
    PARAMS: 
    texts: array of texts to perturb
    span_length: to pass into helper generate_perturbations_
    pct: to pass into helper generate_perturbations_
    ceil_pct: to pass into helper generate_perturbations_
    chunk_size: the number of texts to pass into helper generate_perturbations_ each call
    RETURNS perturbed texts
    """
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(generate_perturbations_(texts[i:i + chunk_size], span_length, pct, mask_model, mask_tokenizer, ceil_pct=ceil_pct))
    return outputs


def perturb_texts(data, mask_model, mask_tokenizer, 
                  perturb_pct=0.3, span_length=2, n_perturbations=1, n_perturbation_rounds=1):
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

    perturb_fn = functools.partial(generate_perturbations, span_length=span_length, pct=perturb_pct,
                                   mask_model=mask_model, mask_tokenizer=mask_tokenizer)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
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

if __name__=='__main__':
    perturb_options = ArgumentParser()
    perturb_options.add_argument('infile', help='where to read candidate passages from')
    perturb_options.add_argument('-k', '--k_examples', help='num examples to load from infile', type=int)
    perturb_options.add_argument('-n', '--n_perturbations', help='number of perturbations to perform in experiments', type=int, default=5)
    perturb_options.add_argument('-s', '--span_length', help='span of tokens to mask in candidate passages', type=int, default=2)
    perturb_options.add_argument('-p', '--perturb_pct', help='percentage (as decimal) of each passage to perturb', type=float, default=0.15)
    perturb_options.add_argument('-r', '--n_perturbation_rounds', help='number of times to attempt perturbations', type=int, default=1)
    perturb_options.add_argument('-w', '--writefile', help='file to write perturbed examples to')

    args = perturb_options.parse_args()

    data = load_data(args.infile, args.k_examples)
    print('Loading T5-3B mask model and tokenizer...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('t5-3b') # can be cached. 
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-3b', model_max_length=n_positions)
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    mask_model.to(DEVICE)
    print('DONE')
    perturbed = perturb_texts(data, mask_model, mask_tokenizer, args.perturb_pct, args.span_length, args.n_perturbations, args.n_perturbation_rounds)
    write_perturbed(perturbed, args.writefile)
