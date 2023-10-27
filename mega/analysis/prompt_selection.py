import os
import argparse
import sys
import time
import random
import json
import wandb
from tqdm import tqdm
import numpy as np
from promptsource.templates import Template, DatasetTemplates
from mega.data.load_datasets import (
    load_xnli_dataset,
    load_xnli_translate_test,
    load_xcopa_dataset,
)
from mega.data.data_utils import choose_few_shot_examples
from mega.eval.eval_cls import evaluate_model
from mega.prompting.prompting_utils import load_prompt_template
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_openai_env_variables
from mega.prompting.create_lang_prompts import add_prompt_to_dataset


def add_prompts_to_lang_xnli(lang, prompt_name):
    prompt_template = load_prompt_template("en", prompt_name, dataset="xnli")
    tgt_prompt_dataset = DatasetTemplates(f"xnli/{lang}")
    add_prompt_to_dataset(
        tgt_prompt_dataset,
        prompt_template,
        lang,
        "en",
        translate=False,
    )


def add_prompts_to_lang_xcopa(lang, prompt_name):
    prompt_template = load_prompt_template("en", prompt_name, dataset="xcopa")
    tgt_prompt_dataset = DatasetTemplates(f"xcopa/{lang}")
    add_prompt_to_dataset(
        tgt_prompt_dataset,
        prompt_template,
        lang,
        "en",
        translate=False,
    )


def main():
    load_env("gpt4v2")
    model = "gpt-35-turbo"

    MAX_VAL_SIZE = 500
    K = 8
    TEMPERATURE = 0

    # XNLI
    xnli_results_file = "mega/analysis/results/prompt_selection_xnli.json"
    if not os.path.exists(xnli_results_file):
        langs = ["sw", "ur"]
        lang2train_dataset = {
            lang: load_xnli_dataset(lang, split="train") for lang in langs
        }

        lang2val_dataset = {
            lang: load_xnli_dataset(lang, split="validation").select(
                list(range(MAX_VAL_SIZE))
            )
            for lang in langs
        }

        lang2test_dataset = {
            lang: load_xnli_dataset(lang, split="test") for lang in langs
        }

        prompt_names = [
            "take the following as truth",
            "does this imply",
            "GPT-3 style",
            "based on the previous passage",
            "guaranteed true",
            "should assume",
            "must be true",
            "can we infer",
            "justified in saying",
            "claim true/false/inconclusive",
            "consider always/sometimes/never",
            "always/sometimes/never",
            "guaranteed/possible/impossible",
            "MNLI crowdsource",
        ]

        # for lang in langs:
        #     for prompt_name in prompt_names:
        #         add_prompts_to_lang_xnli(lang, prompt_name)

        lang2prompt2acc = {}

        for lang in tqdm(langs):
            lang2prompt2acc[lang] = {}
            for prompt_name in tqdm(prompt_names):
                prompt_template = load_prompt_template(
                    lang, prompt_name, dataset="xnli"
                )
                acc = evaluate_model(
                    train_dataset=lang2train_dataset[lang],
                    test_dataset=lang2val_dataset[lang],
                    train_prompt_template=prompt_template,
                    test_prompt_template=prompt_template,
                    model=model,
                    few_shot_size=K,
                    selection_criteria="random",
                    chat_prompt=True,
                    instruction=INSTRUCTIONS.get("xnli", ""),
                    save_preds_path=None,
                    num_evals_per_sec=2,
                    temperature=TEMPERATURE,
                )
                lang2prompt2acc[lang][prompt_name] = acc

        print(lang2prompt2acc)

        with open(xnli_results_file, "w") as f:
            json.dump(lang2prompt2acc, f)

    # XCOPA
    print("XCOPA")
    xcopa_results_file = "mega/analysis/results/prompt_selection_xcopa.json"
    langs = ["ht", "ta"]
    langs = ["ta"]
    lang2train_dataset = {
        lang: load_xcopa_dataset(lang, split="train") for lang in langs
    }
    lang2val_dataset = {
        lang: load_xcopa_dataset(lang, split="validation") for lang in langs
    }
    lang2test_dataset = {lang: load_xcopa_dataset(lang, split="test") for lang in langs}
    if not os.path.exists(xcopa_results_file):
        prompt_names = [
            "C1 or C2? premise, so/because\u2026",
            "best_option",
            "best_option discrete",
            "cause_effect",
            "choose",
            "exercise",
            "i_am_hesitating",
            "more likely",
            "plausible_alternatives",
            "plausible_alternatives_discrete",
        ]

        for lang in langs:
            for prompt_name in prompt_names:
                add_prompts_to_lang_xcopa(lang, prompt_name)

        lang2prompt2acc = {}

        for lang in tqdm(langs):
            lang2prompt2acc[lang] = {}
            for prompt_name in tqdm(prompt_names):
                prompt_template = load_prompt_template(
                    lang, prompt_name, dataset="xcopa"
                )
                acc = evaluate_model(
                    train_dataset=lang2train_dataset[lang],
                    test_dataset=lang2val_dataset[lang],
                    train_prompt_template=prompt_template,
                    test_prompt_template=prompt_template,
                    model=model,
                    few_shot_size=K,
                    selection_criteria="random",
                    chat_prompt=True,
                    instruction=INSTRUCTIONS.get("xcopa", ""),
                    save_preds_path=None,
                    num_evals_per_sec=2,
                    temperature=TEMPERATURE,
                )
                lang2prompt2acc[lang][prompt_name] = acc
        print(lang2prompt2acc)

        with open(xcopa_results_file, "w") as f:
            json.dump(lang2prompt2acc, f)

    else:
        with open(xcopa_results_file) as f:
            lang2prompt2acc = json.load(f)

    # Get best prompts for each language and evaluate on test data using them
    breakpoint()
    test_accs = {}
    for lang in langs:
        best_prompt = max(lang2prompt2acc[lang], key=lang2prompt2acc[lang].get)
        print(f"Best prompt for {lang}: {best_prompt}")
        prompt_template = load_prompt_template(lang, best_prompt, dataset="xcopa")
        acc = evaluate_model(
            train_dataset=lang2train_dataset[lang],
            test_dataset=lang2test_dataset[lang],
            train_prompt_template=prompt_template,
            test_prompt_template=prompt_template,
            model=model,
            few_shot_size=K,
            selection_criteria="random",
            chat_prompt=True,
            instruction=INSTRUCTIONS.get("xcopa", ""),
            save_preds_path=None,
            num_evals_per_sec=2,
            temperature=TEMPERATURE,
        )
        print(f"Test accuracy: {acc}")
        test_accs[lang] = acc

    xcopa_test_results_file = (
        "mega/analysis/results/prompt_tuned_xcopa_test_results.json"
    )
    with open(xcopa_test_results_file, "w") as f:
        json.dump(test_accs, f)


if __name__ == "__main__":
    main()
