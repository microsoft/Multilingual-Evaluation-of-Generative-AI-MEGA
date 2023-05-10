import sys
import glob
import json
import ast
import string
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm
import unicodedata
from functools import partial
from mega.utils.parser import parse_args

PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
MIXED_SEGMENTATION_LANGS = ['zh']

def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer_mlqa(lang, s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text # Hindi does not have formal articles
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'zh':
            return text # Chinese does not have formal articles
        else:
            raise Exception('Unknown Language {}'.format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(PUNCT) #set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, normalize_fn = normalize_answer):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, normalize_fn = normalize_answer):
    return (normalize_fn(prediction) == normalize_fn(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, normalize_fn = normalize_answer):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, normalize_fn)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_results(filename):
    with open(filename) as f:
        results = json.load(f)
    
    print(f"Getting results for {results['tgt_lang']}")
    
    prompt_setting = ""
    if results["pivot_lang"] == results["tgt_lang"]:
        prompt_setting = "Monolingual"
    else:
        if results["translate_test"]:
            prompt_setting = "Translate Test"
        else:
            prompt_setting = "Zero-Shot Cross Lingual"
            
    if results["dataset"] == "indicqa":
        exact_match = results["metrics"]["exact"]
        f1 = results["metrics"]["f1"]
        
    else:
        exact_match = results["metrics"]["exact_match"]
        f1 = results["metrics"]["f1"]

    # preds_path = "/".join(filename.split("/")[:-1])
    # preds_filename = f"{preds_path}/preds.csv"
    # preds_df = pd.read_csv(preds_filename)
    
    # if results["dataset"] == "indicqa":
    #     print(f"Initial Size: {len(preds_df)}")
    #     preds_df = preds_df[preds_df["Label"].apply(lambda x : ast.literal_eval(x)["answers"]["text"][0] != "")]
    #     print(f"Final Size: {len(preds_df)}")
    #     results["metrics"]["f1"] = preds_df["F1-Score"].mean()
    #     results["metrics"]["exact_match"] = preds_df["EM"].mean()
    
    # labels = preds_df["Label"].values
    # preds = preds_df["Prediction"].values
    # exact_match = 0
    # f1 = 0
    # total = len(labels)
    # normalize_answer_mlqa_fn = partial(normalize_answer_mlqa, results['tgt_lang'] if prompt_setting != "Translate Test" else "en")
    # normalize_fn = normalize_answer if results["dataset"] != "mlqa" else normalize_answer_mlqa_fn
    # for pred,label in zip(preds, labels):
    #     pred = ast.literal_eval(pred)
    #     label = ast.literal_eval(label)
    #     ground_truths = label["answers"]["text"]#list(map(lambda x: x['text'], label['answers']))
    #     prediction = pred["prediction_text"]
    #     exact_match += metric_max_over_ground_truths(
    #             exact_match_score, prediction, ground_truths, normalize_fn = normalize_fn)
    #     f1 += metric_max_over_ground_truths(
    #             f1_score, prediction, ground_truths, normalize_fn = normalize_fn)
        
    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total
    
    # print(f"F1 was off by {f1 - results['metrics']['f1']}")
    # print(f"EM was off by {exact_match - results['metrics']['exact_match']}")
        
        
    return {
        "Model" : results["model"],
        "Language" : results["tgt_lang"].split("-")[-1],
        "Prompt Setting": prompt_setting,
        "Prompt Type": results["tgt_prompt_name"],
        "# Few-shot Examples": results["few_shot_k"],
        "Is Dev" : results["eval_on_val"],
        'Test Fraction': results['test_frac'],
        "Short-Contexts": results["short_contexts"],
        "Exact Match": exact_match,
        "F1 Score": f1,
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
    results_final.to_csv(f"results/{args.dataset}_{args.model}_final.csv")
