def random_perturb(texts):
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