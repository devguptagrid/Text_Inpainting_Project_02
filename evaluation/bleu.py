# Computes BLEU score to evaluate the quality of generated text against reference sentences.

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_masked_bleu(reference_ids, generated_ids, mask_positions, tokenizer):
    """
    Compute BLEU only on masked tokens.
    """

    reference_tokens = []
    generated_tokens = []

    for ref_id, gen_id, is_mask in zip(reference_ids, generated_ids, mask_positions): ## Iterate through each token in the reference and generated sequences along with the corresponding mask position to identify which tokens were masked during training and should be considered for BLEU evaluation.
        if is_mask: ## If the current token position is masked, we convert the token IDs to their corresponding tokens using the tokenizer and append them to the reference_tokens and generated_tokens lists, which will be used for BLEU score computation.
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