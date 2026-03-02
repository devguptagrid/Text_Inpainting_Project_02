def lcs(X, Y):
    m = len(X)
    n = len(Y)

    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]


def compute_masked_rouge_l(reference_ids, generated_ids, mask_positions, tokenizer):

    reference_tokens = []
    generated_tokens = []

    for ref_id, gen_id, is_mask in zip(reference_ids, generated_ids, mask_positions):
        if is_mask:
            reference_tokens.append(tokenizer.convert_ids_to_tokens(int(ref_id)))
            generated_tokens.append(tokenizer.convert_ids_to_tokens(int(gen_id)))

    if len(reference_tokens) == 0:
        return 0.0

    lcs_len = lcs(reference_tokens, generated_tokens)

    precision = lcs_len / len(generated_tokens)
    recall = lcs_len / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    rouge_l = (2 * precision * recall) / (precision + recall)

    return rouge_l