#!/usr/bin/env python
# coding: utf-8


import os
import re
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import openai
from datasets import load_dataset
from mega.models.completion_models import gpt3x_completion
from mega.utils.env_utils import load_openai_env_variables


load_openai_env_variables


model = "gpt-35-turbo"


MAX_VAL_SIZE = 500
K = 4
TEMPERATURE = 0


def construct_cot_prompt(
    test_example,
    include_cot=True,
    system_prompt_role="system",
):
    def construct_cot_prompt_example(example, return_out=True):
        premise_templized = ""
        premise = example["premise"]
        if example["question"] == "effect":
            premise_templatized = (
                f'What might have happpened as a result of "{premise}"?'
            )
        else:
            premise_templatized = f'What might be the cause of "{premise}"?'

        # options_templatized = (
        #     f'Options:\n-"{example["choice1"]}"\n-"{example["choice2"]}"'
        # )
        options_templatized = (
            f'Options:\n0."{example["choice1"]}"\n1."{example["choice2"]}"'
        )
        inp = {
            "role": "user",
            "content": f"{premise_templatized}\n{options_templatized}",
        }
        if return_out:
            # answer = (
            #     f"{example['choice1']}"
            #     if example["label"] == 0
            #     else f"{example['choice2']}"
            # )
            answer = str(example["label"])

            if not include_cot:
                out = {"role": "assistant", "content": f'"{answer}"'}
            else:
                out = {
                    "role": "assistant",
                    "content": f"{example['cot']} Therefore, the answer is: \"{answer}\"",
                }
            return [inp, out]

        else:
            return [inp]

    fs_examples_with_cot = [
        {
            "premise": "Adam piyangoyu kazandı.",
            "question": "effect",
            "choice1": "Borçlandı.",
            "choice2": "Zengin oldu.",
            "label": 1,
            "cot": """ Let's think step by step.
The premise "Adam piyangoyu kazandı." can be translated from Turkish into English as "The man won the lottery."
The first option "Borçlandı." can be translated as "He owes money.", whereas the second option "Zengin oldu." can be translated as "He 
became rich."
If the man won the lo"ery, then it makes sense that he became rich as a result.""",
        },
        {
            "premise": "厨师的眼睛流泪了。",
            "question": "cause",
            "choice1": "他切了洋葱。",
            "choice2": "他没有洋葱了。",
            "label": 0,
            "cot": """ Let's think step by step.
The premise "厨师的眼睛流泪了。" can be translated from Mandarin Chinese into English as "The chef's eyes filled with tears."
The first option "他切了洋葱。" can be translated as "He chopped onions.", whereas the second option "他没有洋葱了。" can be translated 
as "He had run out of onions."
It makes sense that the chef's eyes #lled with tears because he chopped onions""",
        },
        {
            "premise": "Warmiqa wasi qhatuqwan huñukurqan.",
            "question": "effect",
            "choice1": "Warmiqa wasita rantinanpaqmi yuyaychakurqan.",
            "choice2": "Warmiqa wasintam pichayta munarqan",
            "label": 0,
            "cot": """Let's think step by step.
The premise "Warmiqa wasi qhatuqwan huñukurqan." can be translated from Cusco-Collao Quechua into English as "The woman called a 
real estate agent."
The first option "Warmiqa wasita rantinanpaqmi yuyaychakurqan." can be translated as "The woman plans to buy a condo.", whereas the 
second option "Warmiqa wasintam pichayta munarqan." can be translated as "The woman needs to clean her house."
If the woman called a real estate agent, then it makes sense that the woman plans to buy a condo as a result.""",
        },
    ]

    prompt = []
    prompt.append(
        {
            "role": system_prompt_role,
            "content": f"Given a premise and a prompt, select the more meaningful of the two choices.",
        }
    )
    for fs_example in fs_examples_with_cot:
        prompt += construct_cot_prompt_example(fs_example, return_out=True)

    prompt += construct_cot_prompt_example(test_example, return_out=False)
    label = str(test_example["label"])

    return prompt, label


langs = ["ht", "ta"]
include_cot = False
system_prompt_role = "user"
lang2score = {}
out_dir = "analysis/results/COT/xcopa/{}"
valid_labels = ["1", "2"]
for lang in langs:
    for seed in [42]:
        out_dir = f"analysis/results/COT/xcopa/{model}/{lang}/explanations{include_cot}_system_prompt_role{system_prompt_role}_seed{seed}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        preds_w_exp = []
        preds_wo_exp = []
        labels = []
        matches = []
        test_dataset = load_dataset("xcopa", lang)["test"]
        pbar = tqdm(test_dataset)
        for test_example in pbar:
            num_fs = K
            while True:
                try:
                    prompt, label = construct_cot_prompt(
                        test_example,
                        include_cot=include_cot,
                        system_prompt_role=system_prompt_role,
                    )
                    out = gpt3x_completion(
                        prompt,
                        model=model,
                        max_tokens=100,
                        temperature=TEMPERATURE,
                        timeout=30,
                    )
                    break
                except openai.error.Timeout:
                    if num_fs >= 0:
                        num_fs -= 1
                        print(
                            f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {(num_fs)}"
                        )
                    else:
                        print("Exausted Everything! Giving Random Prediction Now :(")
                        out = np.random.choice(valid_labels)
                        break
            preds_w_exp.append(out)
            ans_matches = re.findall(r'"([^"]*)"', out)
            # Extract the last match
            if ans_matches:
                pred = ans_matches[-1]
            else:
                pred = ""
            preds_wo_exp.append(pred)
            labels.append(label)
            matches.append(int(preds_wo_exp[-1] == labels[-1]))
            running_acc = np.mean(matches)
            pbar.set_description(f"Accuracy: {running_acc}")

        results_df = pd.DataFrame(
            {
                "Label": labels,
                "Prediction": preds_wo_exp,
                "Match": matches,
                "Prediction With Explanation": preds_w_exp,
            }
        )
        acc = np.mean(matches)

        results_df.to_csv(f"{out_dir}/predictions.csv")
        results_dict = {
            "model": model,
            "dataset": "xcopa",
            "lang": lang,
            "include_cot": include_cot,
            "system_prompt_role": system_prompt_role,
            "seed": seed,
            "metrics": {"accuracy": acc},
        }
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=False)
