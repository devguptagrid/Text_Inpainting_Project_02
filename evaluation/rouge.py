## Computes ROUGE scores to measure overlap between generated text and ground truth.
##ROUGE-L is based on the Longest Common Subsequence (LCS) between two sequences.
def lcs(X, Y): ## Uses dynamic programming to compute the LCS length and return the length of LCS
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


def compute_masked_rouge_l(reference_ids, generated_ids, mask_positions, tokenizer): ##This function calculates ROUGE-L only on masked tokens, which is appropriate for text inpainting tasks.

    reference_tokens = []
    generated_tokens = []

    for ref_id, gen_id, is_mask in zip(reference_ids, generated_ids, mask_positions):
        if is_mask: ##Only positions where mask_positions == True are evaluated.
            reference_tokens.append(tokenizer.convert_ids_to_tokens(int(ref_id))) ##Convert token IDs to text tokens using the tokenizer and append to the reference_tokens and generated_tokens lists for ROUGE-L computation.
            generated_tokens.append(tokenizer.convert_ids_to_tokens(int(gen_id)))

    if len(reference_tokens) == 0:
        return 0.0

    lcs_len = lcs(reference_tokens, generated_tokens) ## Measures overlap between generated and reference tokens.

    precision = lcs_len / len(generated_tokens) ##Compute ROUGE-L precision
    recall = lcs_len / len(reference_tokens) ##Compute ROUGE-L recall

    if precision + recall == 0:
        return 0.0

    rouge_l = (2 * precision * recall) / (precision + recall) ## Compute ROUGE-L F1 score

    return rouge_l