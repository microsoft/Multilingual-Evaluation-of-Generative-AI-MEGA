import os
import sys
import random
import time
import json
from typing import List
import numpy as np
import pandas as pd
import spacy

import unicodedata
from functools import partial
from datasets import load_dataset
import string
import re

from tqdm import tqdm
import openai
from evaluate import load

import numpy as np
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.prompting_utils import construct_langchain_qa_prompt
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_openai_env_variables


load_openai_env_variables()

PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]


TYDIQA_LANG2CODES = {
    "bengali": "bn",
    "korean": "ko",
    "swahili": "sw",
    "english": "en",
    "indonesian": "id",
    "arabic": "ar",
    "finnish": "fi",
    "telugu": "te",
    "russian": "ru",
}

langcodes2lang = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "hi": "Hindi",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin",
}


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
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
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        elif lang == "hi":
            return text  # Hindi does not have formal articles
        elif lang == "vi":
            return re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", text)
        elif lang == "de":
            return re.sub(
                r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
                " ",
                text,
            )
        elif lang == "ar":
            return re.sub("\sال^|ال", " ", text)
        elif lang == "zh":
            return text  # Chinese does not have formal articles
        else:
            raise Exception("Unknown Language {}".format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception("Unknown Language {}".format(lang))
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(PUNCT)  # set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


PROMPTS_DICT = {
    "answer_given_context_and_question": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    "answer_given_context_and_question+unaswerable": """{context}
    Q: {question}

    Referring to the passage above, what will be the correct answer to the given question? If you can't find the answer, please respond "unanswerable".
    {answer}""",
    "lang_instruct_answer_given_context_and_question": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is? Please try to answer in {language} and ensure that the answer appears as it is in the passage.
    A: {answer}""",
}


class SpacySentenceTokenizer:
    def __init__(self):
        self.nlp = spacy.load("xx_ent_wiki_sm")
        self.nlp.add_pipe("sentencizer")

    def __call__(self, text: str) -> List[str]:
        return list(map(lambda span: span.text, self.nlp(text).sents))


def load_qa_dataset(dataset_name, lang, split, dataset_frac=1, translate_test=False):
    if dataset_name == "indicqa":
        if split != "train":
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")[split]
        else:
            dataset = load_dataset("squad_v2")[split]
    elif dataset_name == "xquad":
        if split != "train":
            dataset = load_dataset("xquad", f"xquad.{lang}")[split]
        else:
            dataset = load_dataset("squad")[split]
    elif dataset_name == "tydiqa":
        dataset = load_dataset("tydiqa", "secondary_task")[split]
        dataset = dataset.map(
            lambda example: {"lang": TYDIQA_LANG2CODES[example["id"].split("-")[0]]}
        )
        dataset = dataset.filter(lambda example: example["lang"] == lang)
    elif dataset_name == "mlqa":
        if split == "train":
            print("No Training Data for MLQA, switching to validation!")
            split = "validation"
        if translate_test:
            dataset_name = f"mlqa-translate-test.{lang}"
        else:
            dataset_name = f"mlqa.{lang}.{lang}"

        dataset = load_dataset("mlqa", dataset_name)[split]

    else:
        raise NotImplementedError()
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def eval_qa(
    test_dataset,
    prompt,
    model,
    num_evals_per_sec=1,
    smaller_prompts=[],
    metric="squad",
    normalize_fn=normalize_answer,
    **model_kwargs,
):
    from mega.models.qa_models import answer_question

    preds = []
    labels = []
    matches = []
    f1_scores = []
    em_score = 0
    f1_score = 0
    squad_metric = load(metric)
    pbar = tqdm(enumerate(test_dataset))
    for i, test_example in pbar:
        prompt_to_use = prompt
        for trial in range(0, len(smaller_prompts) + 1):
            try:
                pred = answer_question(
                    model,
                    test_example["question"],
                    test_example["context"],
                    prompt=prompt_to_use,
                    chunk_size=model_kwargs.get("chunk_size", 100),
                    chunk_overlap=model_kwargs.get("chunk_overlap", 0),
                ).strip()
                break
            except openai.error.InvalidRequestError as e:
                print(e, "Request here")
                if trial == len(smaller_prompts):
                    print("Exausted Everything! Giving Empty Prediction Now :(")
                    pred = ""
                    break
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(smaller_prompts) - trial - 1}"
                )
                prompt_to_use = smaller_prompts[trial]

            except openai.error.APIError as e:
                print("Content Policy Triggered! Giving Empty prediction for this!")
                pred = ""
                break

        pred = normalize_fn(pred)
        if metric == "squad":
            prediction = {"prediction_text": pred, "id": test_example["id"]}
        else:
            no_answer_probability = float("unanswerable" in pred)
            prediction = {
                "prediction_text": pred,
                "id": test_example["id"],
                "no_answer_probability": no_answer_probability,
            }

        # if no_answer_probability == 1.0:
        #     breakpoint()

        reference = {}
        reference["answers"] = test_example["answers"]
        reference["id"] = test_example["id"]
        if reference["answers"]["text"][0] == "":
            reference["answers"]["text"] = []
            reference["answers"]["answer_start"] = []

        if metric == "squad":
            results = squad_metric.compute(
                predictions=[prediction], references=[reference]
            )
        else:
            results = squad_metric.compute(
                predictions=[prediction],
                references=[reference],
                no_answer_threshold=0.9,
            )

        if metric == "squad":
            em_score += results["exact_match"]
            matches.append(results["exact_match"])
        else:
            em_score += results["exact"]
            matches.append(results["exact"])

        f1_scores.append(results["f1"])
        f1_score += results["f1"]
        preds.append(prediction)
        labels.append(reference)
        time.sleep(1 / num_evals_per_sec)

        avg_f1 = np.mean(f1_scores)
        avg_em = np.mean(matches)

        pbar.set_description(f"em: {avg_em} f1: {avg_f1}. {i+1}/{len(test_dataset)}")

    results_df = pd.DataFrame(
        {"Label": labels, "Prediction": preds, "EM": matches, "F1": f1_scores}
    )

    if metric == "squad":
        metrics = squad_metric.compute(predictions=preds, references=labels)
    else:
        metrics = squad_metric.compute(
            predictions=preds, references=labels, no_answer_threshold=0.9
        )

    return metrics, results_df


def main():
    args = parse_args(sys.argv[1:])

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = f"{args.save_dir}/{args.dataset}_gptindex/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_FewShotK_{args.few_shot_k}"

    if args.short_contexts:
        out_dir = f"{out_dir}_shortContexts"

    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_dataset = load_qa_dataset(
        args.dataset,
        lang=args.pivot_lang,
        split="train" if not args.use_val_to_prompt else "validation",
    )
    test_dataset = load_qa_dataset(
        args.dataset,
        lang=args.tgt_lang,
        split="validation" if args.eval_on_val else "test",
        dataset_frac=args.test_frac,
        translate_test=args.translate_test,
    )

    train_examples = choose_few_shot_examples(
        train_dataset, args.few_shot_k, args.few_shot_selection
    )

    if args.short_contexts:
        sent_tokenizer = SpacySentenceTokenizer()

        for train_example in train_examples:
            sents = sent_tokenizer(train_example["context"])
            if train_example["answers"]["text"] == []:
                context = sents[0]
            else:
                context = sents[0]
                for sent in sents:
                    if train_example["answers"]["text"][0] in sent:
                        context = sent

        # breakpoint()

    train_prompt_template = PROMPTS_DICT[args.pivot_prompt_name]
    test_prompt_template = PROMPTS_DICT[args.tgt_prompt_name]
    if args.pivot_prompt_name == "lang_instruct_answer_given_context_and_question":
        train_prompt_template = train_prompt_template.replace(
            "{language}", langcodes2lang[args.pivot_lang]
        )
    if args.tgt_prompt_name == "lang_instruct_answer_given_context_and_question":
        test_prompt_template = test_prompt_template.replace(
            "{language}", langcodes2lang[args.tgt_lang]
        )

    langchain_prompt = construct_langchain_qa_prompt(
        train_examples,
        train_prompt_template=train_prompt_template,
        test_prompt_template=test_prompt_template,
    )
    smaller_prompts = []
    for i in range(1, args.few_shot_k + 1):
        smaller_prompts.append(
            construct_langchain_qa_prompt(
                train_examples[: args.few_shot_k - i],
                train_prompt_template=train_prompt_template,
                test_prompt_template=test_prompt_template,
            )
        )
    normalize_answer_mlqa_fn = partial(
        normalize_answer_mlqa, args.tgt_lang if not args.translate_test else "en"
    )
    normalize_fn = (
        normalize_answer if args.dataset != "mlqa" else normalize_answer_mlqa_fn
    )

    metrics, results_df = eval_qa(
        test_dataset,
        langchain_prompt,
        args.model,
        num_evals_per_sec=args.num_evals_per_sec,
        smaller_prompts=smaller_prompts,
        metric="squad" if args.dataset != "indicqa" else "squad_v2",
        normalize_fn=normalize_fn,
    )

    print(metrics)
    pred_file_path = f"{out_dir}/preds.csv"
    results_df.to_csv(pred_file_path)

    # Store results
    results_dict = vars(args)
    results_dict["metrics"] = metrics
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    # if args.log_wandb:
    #     wandb.log({"accuracy": accuracy})


if __name__ == "__main__":
    main()
