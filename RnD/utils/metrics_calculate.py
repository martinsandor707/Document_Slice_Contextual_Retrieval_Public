import pandas as pd
import numpy as np
import math

def calculate_advanced_retrieval_metrics(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Calculates 1-Recall@k, Context Ratio, MRR@k, and NDCG@k.
    
    Parameters:
    - df: DataFrame containing 'retrieved_hashes' and 'supporting_hashes'.
    - k: The rank cutoff for evaluation.
    """
    
    # --- Pre-computation for optimization ---
    # Log discounts for NDCG to avoid re-calculating log2 for every row
    discounts = 1 / np.log2(np.arange(2, k + 2))

    def calculate_row_metrics(row):
        retrieved = row['retrieved_hashes']
        ground_truth = row['supporting_hashes']
        
        # 1. Handling Edge Cases (No ground truth)
        if not ground_truth or len(ground_truth) == 0:
            return pd.Series({
                f'1_minus_recall_at_{k}': 0.0,
                'context_precision_ratio': 0.0,
                f'mrr_at_{k}': 0.0,
                f'ndcg_at_{k}': 0.0
            })
            
        # Data Prep
        # Truncate retrieved to top-k
        retrieved_k = retrieved[:k]
        ground_truth_set = set(ground_truth)
        
        # Create a binary relevance vector for the top k
        # rel_vector[i] is 1 if retrieved_k[i] is relevant, else 0
        rel_vector = [1 if h in ground_truth_set else 0 for h in retrieved_k]
        
        # --- Metric A: 1 - Recall@k ---
        relevant_retrieved_count = sum(rel_vector)
        total_relevant = len(ground_truth_set)
        recall = relevant_retrieved_count / total_relevant
        miss_rate = 1.0 - recall

        # --- Metric B: Context Ratio (Relevant / Irrelevant) ---
        # Note: This usually considers the whole context window, but consistent with @k here
        # If you need full window, remove the [:k] slice for this specific metric.
        irrelevant_count = len(retrieved_k) - relevant_retrieved_count
        if irrelevant_count == 0:
            context_ratio = np.inf if relevant_retrieved_count > 0 else 0.0
        else:
            context_ratio = relevant_retrieved_count / irrelevant_count

        # --- Metric C: MRR@k ---
        mrr = 0.0
        for idx, is_rel in enumerate(rel_vector):
            if is_rel:
                mrr = 1.0 / (idx + 1)
                break # Stop at the first relevant item

        # --- Metric D: NDCG@k ---
        # 1. DCG (Actual)
        # We zip the binary relevance with the pre-computed log discounts
        dcg = sum([rel * disc for rel, disc in zip(rel_vector, discounts[:len(rel_vector)])])
        
        # 2. IDCG (Ideal)
        # The ideal scenario is that the top 'total_relevant' items are all 1s.
        # We are bounded by k.
        ideal_count = min(total_relevant, k)
        idcg = sum(discounts[:ideal_count])
        
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return pd.Series({
            f'1_minus_recall_at_{k}': miss_rate,
            'context_precision_ratio': context_ratio,
            f'mrr_at_{k}': mrr,
            f'ndcg_at_{k}': ndcg
        })

    # Apply to dataframe
    results_df = df.copy()
    metrics_columns = results_df.apply(calculate_row_metrics, axis=1)
    
    # Concatenate original data with new metrics
    return metrics_columns

# --- Verification ---
data = {
    'question_id': [1, 2],
    'retrieved_hashes': [
        ['a1', 'b2', 'c3', 'd4'], # Row 1: Rel at index 0 and 2
        ['x9', 'y8']              # Row 2: No relevant
    ],
    'supporting_hashes': [
        ['a1', 'c3', 'z99'],      # Row 1
        ['a1']                    # Row 2
    ]
}
df_test = pd.DataFrame(data)
df_test = df_test.set_index('question_id')

df_final = calculate_advanced_retrieval_metrics(df_test, k=4)

print(df_final)