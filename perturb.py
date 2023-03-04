import numpy as np
import tqdm
import transformers
import re
from torch import cuda

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MASK_PATTERN = re.compile(r"<extra_id_\d+>")


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
