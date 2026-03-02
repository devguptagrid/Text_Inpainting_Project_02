from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_masked_bleu(reference_ids, generated_ids, mask_positions, tokenizer):
    """
    Compute BLEU only on masked tokens.
    """

    reference_tokens = []
    generated_tokens = []

    for ref_id, gen_id, is_mask in zip(reference_ids, generated_ids, mask_positions):
        if is_mask:
            reference_tokens.append(tokenizer.convert_ids_to_tokens(int(ref_id)))
            generated_tokens.append(tokenizer.convert_ids_to_tokens(int(gen_id)))

    if len(reference_tokens) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1

    score = sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=smoothing
    )

    return score