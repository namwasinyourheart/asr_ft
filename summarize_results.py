import os
import re
import pandas as pd

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["Experiment Name", "Experiment Variant", "Results File", "Overall WER"]

    base_cols = [c for c in base_cols if c in df.columns]

    dialect_cols = [c for c in df.columns if c.startswith("dialect_")]
    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    province_cols = [c for c in df.columns if c.startswith("province_name_")]

    ordered_cols = base_cols + dialect_cols + gender_cols + province_cols
    other_cols = [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols + other_cols]


def summarize_results(exps_dir="exps/", output_csv="results_summary.csv"):
    """
    Reads results from '{exps_dir}/{exp_name}/{exp_variant}/results/test_metrics*.txt' files, 
    extracts WER and group-level WERs, saves them to a CSV file, and displays them in Markdown format.
    """
    experiments = []
    results_dir = os.path.join(exps_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for exp_name in os.listdir(exps_dir):
        exp_path = os.path.join(exps_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        for exp_variant in os.listdir(exp_path):
            exp_results_dir = os.path.join(exp_path, exp_variant, "results")
            if not os.path.isdir(exp_results_dir):
                continue

            for results_file in os.listdir(exp_results_dir):
                if results_file.startswith("test_metrics") and results_file.endswith(".txt"):
                    results_file_path = os.path.join(exp_results_dir, results_file)

                    with open(results_file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Parse WER
                    wer_match = re.search(r"wer:\s*([\d\.]+)", content)
                    wer = float(wer_match.group(1)) if wer_match else None

                    # Parse wer_by_group (dictionary string)
                    group_match = re.search(r"wer_by_group:\s*(\{.*\})", content, re.DOTALL)
                    group_dict = {}
                    if group_match:
                        try:
                            group_dict = eval(group_match.group(1))  # file này là trusted
                        except Exception:
                            group_dict = {}

                    row = {
                        "Experiment Name": exp_name,
                        "Experiment Variant": exp_variant,
                        "Results File": results_file,
                        "Overall WER": wer,
                    }

                    # Flatten group metrics
                    for group_key, group_vals in group_dict.items():
                        for k, v in group_vals.items():
                            k_str = str(k)
                            k_str = k_str.replace("dialect", "")
                            k_str = k_str.strip().replace(" ", "_") 
                            row[f"{group_key}_{k_str}"] = v

                            # print(row)
                            # break
                        # break

                    experiments.append(row)

    df_results = pd.DataFrame(experiments)

    if not df_results.empty:
        df_results = reorder_columns(df_results)

    output_path = os.path.join(results_dir, output_csv)
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # if not df_results.empty:
    #     markdown_table = df_results.to_markdown(index=False)
    #     print("\n📊 Experiment Results\n")
    #     print(markdown_table)

    return df_results

if __name__ == "__main__":
    summarize_results(exps_dir="exps", output_csv="results_summary.csv")
                
# Example Usage
# df_results = summarize_results(exps_dir="exps", output_csv="results_summary.csv")