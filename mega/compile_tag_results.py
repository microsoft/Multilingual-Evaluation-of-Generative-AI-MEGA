import sys
import glob
import json
import pandas as pd
from tqdm import tqdm
from mega.utils.parser import parse_args

def get_results(filename):
    with open(filename) as f:
        results = json.load(f)
    prompt_setting = ""
    if results["pivot_lang"] == results["tgt_lang"]:
        prompt_setting = "Monolingual"
    else:
        if results["translate_test"] or "translation" in results["tgt_lang"]:
            prompt_setting = "Translate Test"
        else:
            prompt_setting = "Zero-Shot Cross Lingual"
    return {
        "Model": results["model"],
        "Language": results["tgt_lang"].split("-")[-1],
        "Prompt Setting": prompt_setting,
        "Prompt Type": results["tgt_prompt_name"],
        "# Few-shot Examples": results["few_shot_k"],
        "F1": results["metrics"]["f1-score"],
        "Is Dev" : results["eval_on_val"],
        'Test Fraction': results['test_frac']
    }


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.dataset == "paws-x":
        args.dataset = "pawsx"
    filenames = glob.glob(f"results/{args.dataset}/{args.model}/**/**/*.json")
    result_rows = []
    for filename in tqdm(filenames):
        result_rows.append(get_results(filename))

    results_final = pd.DataFrame(result_rows)
    results_final.to_csv(f"results/{args.dataset}_{args.model}_final_.csv")

