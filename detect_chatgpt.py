"""
Project model.

TASKS: 
1. correct+finish get_perturbation_results
2. correct+finish main code, linking armaan's data gen and probability code (call armaan's data generating functions in main() script here, pass data into get_perturbation_results)
3. ...theoretically things should run after that

Optional: 
- understand: replace_masks, extract_fills, apply_extracted_fills, perturb_texts_
"""

import transformers
import functools
import re

import detect_gpt

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
mask_pattern = re.compile(r"<extra_id_\d+>")
mask_filling_model_name = "t5-3b"

# DESC: loads and returns mask model. 
# CALLED BY: get_perturbation_results
def load_mask_model():
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name) # can be cached. 
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)
    
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()
    mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

    return mask_model, mask_tokenizer

# DESC: masks portions of a given text. 
# RETURNS: the text, with randomly chosen word sequences masked with "<extra_id_0>" instead, where the last 
# number increments for each masked sequence
# CALLED BY: perturb_texts_
# PARAMS: 
# span_length: the number of words in a "span", i.e. a single masked sequence
# pct: the percent of text that should be perturbed (percentage of words)
# ceil_pct: round number of masked sequences up if true, down if false. 
# buffer_size: the number of unmasked words that must be between each masked sequence
def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
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

# DESC: counts the number of masks in each text in texts.  
# RETURNS: an array of numbers, each of which is the number of masks for the corresponding text in texts
# CALLED BY: replace_masks
# PARAMS: 
# texts: an array of texts. 
def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# !!!RELIES ON MASK_MODEL AND MASK_TOKENIZER AS GLOBAL VARIABLE!!!
# DESC: return a sample from T5 mask_model for each masked span
# CALLED BY: perturb_texts_
# PARAMS: 
# texts: an array of texts
# TODO unchecked function
def replace_masks(texts):
    n_expected = count_masks(texts) # number of masks in texts
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0] # the integer id corresponding to the masking string's token
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE) # tokenize the texts
    # START unchecked function
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1, num_return_sequences=1, eos_token_id=stop_id)
    # END unchecked function
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False) # convert tokenized texts (list of integer ids) back to text strings

# DESC: return the text without the mask tokens
# CALLED BY: perturb_texts_
# TODO unchecked function
def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [mask_pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


# DESC: insert the generated text into the pre-existing text 
# CALLED BY: perturb_texts_
# TODO unchecked function
def apply_extracted_fills(masked_texts, extracted_fills):
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

# DESC: helper function for perturb_texts
# CALLED BY: perturb_texts
# TODO unchecked function
def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    # START unchecked function
    raw_fills = replace_masks(masked_texts)
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
    # fairly certain below is unnecessary (replaces mask with random words from dictionary)
    '''
    if args.random_fills:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text
    '''
    # END unchecked function
    return perturbed_texts

# DESC: perturbs the texts. 
# CALLED BY: get_perturbation_results
# PARAMS: 
# texts: array of texts to perturb
# span_length: to pass into helper perturb_texts_
# pct: to pass into helper perturb_texts_
# ceil_pct: to pass into helper perturb_texts_
# chunk_size: the number of texts to pass into helper perturb_texts_ each call
def perturb_texts(texts, span_length, pct, ceil_pct=False, chunk_size=20):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


# TODO: unchecked function. 
# DESC: return likelihoods from perturbed samples
def get_perturbation_results(data, pct_words_masked=0.3, span_length=2, n_perturbations=1, n_samples=500):
    mask_model, mask_tokenizer = load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=pct_words_masked)

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

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results
    # END unchecked function



if __name__ == '__main__':
    
    data = "" # TODO 
    perturbation_results = get_perturbation_results(data)

    # START unchecked code
    for perturbation_mode in ['d', 'z']:
    output = run_perturbation_experiment(
        perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
    outputs.append(output)
    with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
        json.dump(output, f)
    # END unchecked code