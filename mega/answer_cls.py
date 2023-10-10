import os
import argparse
import sys
import time
import random
import json
import wandb
import numpy as np
import spacy
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from promptsource.templates import Template, DatasetTemplates
from mega.utils.parser import parse_args
from mega.models.completion_models import model_completion
from mega.data.data_utils import choose_few_shot_examples
import pdb

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

# PROMPT_TEMPLATE = """Based on the question and context below
# Q: {question_str}
# C: {context_str}
# Answer if the answer to the question can be present in the context. Yes or No?
# A:"""

# PROMPT_TEMPLATE = """Question: {question_str}
# Context: {context_str}
# Can the question be answered based on the context? Yes or No?
# """

# PROMPT_TEMPLATE = """Can you answer the question "{question_str}" based only on the following:
# {context_str}
# """

PROMPT_TEMPLATE = """{context_str}
Does that sentence have all you need to answer the question "{question_str}"
"""


class SpacySentenceTokenizer:
    def __init__(self):
        self.nlp = spacy.load("xx_ent_wiki_sm")
        self.nlp.add_pipe("sentencizer")

    def __call__(self, text: str) -> List[str]:
        return list(map(lambda span: span.text, self.nlp(text).sents))


def get_few_shot_samples(
    train_dataset, few_shot_size, selection_criteria, sent_tokenizer, qrpc=False
):
    if qrpc:
        qrpc_dataset = load_dataset("glue", "qnli")["train"]
        fs_examples = choose_few_shot_examples(
            qrpc_dataset, few_shot_size, selection_criteria
        )
        prompt_template = DatasetTemplates("glue/qnli")["have all you need"]
        fs_prompts = [
            "\n".join(prompt_template.apply(example)) for example in fs_examples
        ]
        return "\n\n".join(fs_prompts)

    else:
        fs_examples = choose_few_shot_examples(
            train_dataset, few_shot_size, selection_criteria
        )
        fs_prompt = []
        for example in fs_examples:
            context = example["context"]
            context_sents = sent_tokenizer(context)
            answer = example["answers"]["text"][0]
            question = example["question"]
            random.shuffle(context_sents)
            for sent in context_sents:
                if answer in sent:
                    prompt = PROMPT_TEMPLATE.replace(
                        "{question_str}", question
                    ).replace("{context_str}", sent)
                    prompt += "Yes\n\n"
                    fs_prompt.append(prompt)
                    break
            for sent in context_sents:
                if answer not in sent:
                    prompt = PROMPT_TEMPLATE.replace(
                        "{question_str}", question
                    ).replace("{context_str}", sent)
                    prompt += "No\n\n"
                    fs_prompt.append(prompt)
                    break
        random.shuffle(fs_prompt)
        return "".join(fs_prompt)


def get_line_with_answer(
    example, sent_tokenizer, model, fs_examples="", **model_params
):
    question = example["question"]
    answer = example["answers"]["text"][0]
    context = example["context"]
    context_sents = sent_tokenizer(context)
    context_with_answer = ""
    for sent in context_sents:
        if answer in sent:
            context_with_answer = sent
            break

    # Ask GPT about which line contains the answer
    predicted_context_with_answer = ""
    for sent in context_sents:
        prompt = fs_examples + PROMPT_TEMPLATE.replace(
            "{question_str}", question
        ).replace("{context_str}", sent)
        try:
            response = model_completion(prompt, model, **model_params)
        except openai.error.InvalidRequestError:
            response = np.random.choose(["yes", "no"])

        if response.lower() == "yes" or response.lower() == "yes.":
            predicted_context_with_answer = sent
            break

    return predicted_context_with_answer, int(
        predicted_context_with_answer == context_with_answer
    )


def load_qa_dataset(dataset_name, lang, split, dataset_frac=1):
    if dataset_name == "xquad":
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
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def main(sys_args):
    args = parse_args(sys_args)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = f"{args.save_dir}/{args.dataset}_ac/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_FewShotK_{args.few_shot_k}"

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
    )

    tokenizer = SpacySentenceTokenizer()

    fs_examples = get_few_shot_samples(
        train_dataset, args.few_shot_k // 2, args.few_shot_selection, tokenizer
    )
    # Zero-Shot evaluation
    matches = [
        get_line_with_answer(
            example, tokenizer, model="DaVinci003", fs_examples=fs_examples
        )[1]
        for example in tqdm(test_dataset)
    ]
    acc = np.mean(matches)
    print(acc)

    results_dict = vars(args)
    results_dict["metrics"] = {"accuracy": acc}
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
