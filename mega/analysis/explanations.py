#!/usr/bin/env python
# coding: utf-8

# In[2]:
# In[63]:


import os
import argparse
import sys
import time
import random
import json
import wandb
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
import openai
from mega.data.load_datasets import load_xcopa_dataset
from mega.data.data_utils import choose_few_shot_examples
from mega.eval.eval_cls import evaluate_model
from mega.models.completion_models import gpt3x_completion
from mega.prompting.prompting_utils import load_prompt_template
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_env


# In[30]:


load_env("gpt4v2")


# In[38]:


model = "gpt-35-turbo"


# In[33]:


MAX_VAL_SIZE = 500
K = 4
TEMPERATURE = 0


# In[10]:


langs = ["ht", "ta"]
lang = langs[0]
filename = glob.glob("data/natural-instructions-2.8/tasks/*xcopa*reasoning*ht*")[0]


# In[13]:


with open(filename) as f:
    sni_xcopa = json.load(f)


# In[21]:


sni_xcopa


# In[67]:


def construct_sni_prompt(
    sni_dict,
    test_example,
    include_explanation=True,
    k=4,
    system_prompt_role="system",
):
    def construct_sni_prompt_example(example):
        inp = {"role": "user", "content": example["input"]}
        if include_explanation:
            out = {
                "role": "assistant",
                "content": f"{example['explanation']} Hence the answer is, {example['output']}",
            }
        else:
            out = {"role": "assistant", "content": f"{example['output']}"}

        return [inp, out]

    system_prompt = {
        "role": system_prompt_role,
        "content": f"{sni_dict['Definition'][0]}",
    }
    examples = []
    pos_nd_neg_exs = sni_dict["Positive Examples"] + sni_dict["Negative Examples"]
    random.seed(42)
    random.shuffle(pos_nd_neg_exs)
    for i in range(min(k, len(pos_nd_neg_exs))):
        examples += construct_sni_prompt_example(pos_nd_neg_exs[i])

    test_prompt = {"role": "user", "content": test_example["input"]}

    return [system_prompt] + examples + [test_prompt]


# In[44]:


prompt = construct_sni_prompt(sni_xcopa, sni_xcopa["Instances"][0])
prompt


# In[45]:


out = gpt3x_completion(prompt, model=model, max_tokens=100, temperature=TEMPERATURE)


# In[46]:


out


# In[ ]:


langs = ["ht", "ta"]
include_explanation = True
system_prompt_role = "system"
lang2score = {}
out_dir = "analysis/results/explanations/xcopa/{}"
valid_labels = ["1", "2"]
for lang in langs:
    for seed in [1, 11, 42]:
        out_dir = f"analysis/results/explanations/xcopa/{model}/{lang}/explanations{include_explanation}_system_prompt_role{system_prompt_role}_seed{seed}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename = glob.glob(
            f"data/natural-instructions-2.8/tasks/*xcopa*reasoning*{lang}*"
        )[0]
        with open(filename) as f:
            sni_xcopa = json.load(f)
        preds_w_exp = []
        preds_wo_exp = []
        labels = []
        matches = []
        pbar = tqdm(sni_xcopa["Instances"][:500])
        for test_example in pbar:
            num_fs = K
            while True:
                try:
                    prompt = construct_sni_prompt(
                        sni_xcopa,
                        test_example,
                        include_explanation=include_explanation,
                        system_prompt_role=system_prompt_role,
                        k=num_fs,
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
            preds_wo_exp.append(out.split(",")[-1].strip().split(".")[0])
            labels.append(test_example["output"][0])
            matches.append(preds_wo_exp[-1] == labels[-1])
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
            "include_explanation": include_explanation,
            "system_prompt_role": system_prompt_role,
            "seed": seed,
            "metrics": {"accuracy": acc},
        }
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=False)


# In[69]:


langs = ["ht", "ta"]
include_explanation = False
system_prompt_role = "system"
lang2score = {}
out_dir = "analysis/results/explanations/xcopa/{}"
valid_labels = ["1", "2"]
for lang in langs:
    for seed in [1, 11, 42]:
        out_dir = f"analysis/results/explanations/xcopa/{model}/{lang}/explanations{include_explanation}_system_prompt_role{system_prompt_role}_seed{seed}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename = glob.glob(
            f"data/natural-instructions-2.8/tasks/*xcopa*reasoning*{lang}*"
        )[0]
        with open(filename) as f:
            sni_xcopa = json.load(f)
        preds_w_exp = []
        preds_wo_exp = []
        labels = []
        matches = []
        pbar = tqdm(sni_xcopa["Instances"][:500])
        for test_example in pbar:
            num_fs = K
            while True:
                try:
                    prompt = construct_sni_prompt(
                        sni_xcopa,
                        test_example,
                        include_explanation=include_explanation,
                        system_prompt_role=system_prompt_role,
                        k=num_fs,
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
            preds_wo_exp.append(out.split(",")[-1].strip().split(".")[0])
            labels.append(test_example["output"][0])
            matches.append(preds_wo_exp[-1] == labels[-1])
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
            "include_explanation": include_explanation,
            "system_prompt_role": system_prompt_role,
            "seed": seed,
            "metrics": {"accuracy": acc},
        }
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=False)


# In[ ]:
