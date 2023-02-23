import glob
import json
import pandas as pd
from tqdm import tqdm


def get_results(filename):
    with open(filename) as f:
        results = json.load(f)
    prompt_setting = ""
    if results["pivot_lang"] == results["tgt_lang"]:
        prompt_setting = "Monolingual"
    else:
        if results["translate_test"]:
            prompt_setting = "Translate Test"
        else:
            prompt_setting = "Zero-Shot Cross Lingual"
    return {
        "Model": results["model"],
        "Language": results["tgt_lang"],
        "Prompt Setting": prompt_setting,
        "Prompt Type": results["tgt_prompt_name"],
        "# Few-shot Examples": results["few_shot_k"],
        "Accuracy": results["metrics"]["accuracy"],
    }


if __name__ == "__main__":
    model = "DaVinci003"
    task = "pawsx"
    filenames = glob.glob(f"results/{task}/{model}/**/**/*.json")
    result_rows = []
    for filename in tqdm(filenames):
        result_rows.append(get_results(filename))

    results_final = pd.DataFrame(result_rows)
    results_final.to_csv("results/xnli_final.csv")
