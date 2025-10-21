import argparse
import pandas as pd
import metrics


def evaluate(true_df: pd.DataFrame, pred_df: pd.DataFrame):
    if not {"err_sentence", "cor_sentence"}.issubset(true_df.columns):
        raise ValueError(f"Truth DF must have columns 'err_sentence' and 'cor_sentence' (found: {list(true_df.columns)})")
    if "cor_sentence" not in pred_df.columns:
        raise ValueError(f"Prediction DF must have column 'cor_sentence' (found: {list(pred_df.columns)})")

    # Competition-style strictness: require same length and (if provided) same err_sentence order
    if len(true_df) != len(pred_df):
        raise ValueError(f"Length mismatch: truth={len(true_df)} vs pred={len(pred_df)}. Ensure one-to-one rows.")
    if "err_sentence" in pred_df.columns:
        if not true_df["err_sentence"].astype(str).equals(pred_df["err_sentence"].astype(str)):
            raise ValueError("Row order/content mismatch in 'err_sentence' between truth and submission.")

    pred_df = pd.DataFrame({"cor_sentence": pred_df["cor_sentence"].astype(str)})

    # Get results from metrics
    results = metrics.evaluate_correction(true_df, pred_df)
    
    # Add original_target_part and golden_target_part to analysis_df if they exist
    if "original_target_part" in true_df.columns and "golden_target_part" in true_df.columns:
        results['analysis_df']['original_target_part'] = true_df['original_target_part'].values
        results['analysis_df']['golden_target_part'] = true_df['golden_target_part'].values
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission against truth using metrics.py")
    parser.add_argument("--true_df", default="data/train_dataset.csv", help="Path to ground truth CSV containing err_sentence, cor_sentence")
    parser.add_argument("--pred_df", default="submission.csv", help="Path to submission CSV containing cor_sentence")
    parser.add_argument("--output",  default="analysis.csv", help="Path to save analysis DataFrame as CSV (optional)")
    args = parser.parse_args()

    true_df = pd.read_csv(args.true_df)
    sub_df = pd.read_csv(args.pred_df)

    results = evaluate(true_df, sub_df)
    
    # Export analysis DataFrame if output path is provided
    if args.output:
        results['analysis_df'].to_csv(args.output, index=False)
        print(f"Analysis results saved to {args.output}")


if __name__ == "__main__":
    main()
