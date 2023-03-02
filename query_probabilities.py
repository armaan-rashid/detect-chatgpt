"""
These are the essential detection functions for detectChatGPT, 
adapted from Eric Mitchell et al.'s original implementation. 

TODO: Turn this file into implementation of querying multiple models, and
training an NN to weight multiple models' query methods
"""


import openai
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
import transformers


API_TOKEN_COUNTER = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2')

def get_ll(text, openai_model=None, base_tokenizer=None, base_model=None):
    """
    TODO: edit this function to query multiple models
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
        kwargs = { "engine": openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        assert base_tokenizer is not None and base_model is not None
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, openai_model=None, base_tokenizer=None, base_model=None, batch_size=50):
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
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(batch_size)   # speed things up!
        return pool.map(get_ll, texts)


