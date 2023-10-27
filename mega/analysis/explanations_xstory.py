#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


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
from mega.data.data_utils import choose_few_shot_examples
from mega.eval.eval_cls import evaluate_model
from mega.models.completion_models import gpt3x_completion
from mega.prompting.prompting_utils import load_prompt_template
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_openai_env_variables
from mega.data.load_datasets import load_xstory_cloze_dataset

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import json

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 14})


# In[3]:


load_env("gpt4v3")


# In[4]:


model = "gpt-35-tunro"


# In[5]:


MAX_VAL_SIZE = 500
K = 4
TEMPERATURE = 0


# In[6]:


langs = ["en", "my", "te"]
lang = langs[0]
filename = "data/natural-instructions-2.8/tasks/task296_storycloze_correct_end_classification.json"


# In[7]:


with open(filename) as f:
    sni_storycloze = json.load(f)


# In[8]:


sni_storycloze


# In[9]:


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


# In[10]:


def get_xstorycloze_sni_instances(dataset):
    sni_instances = []

    for example in dataset:
        sni_instance_input = f"Sentence1: {example['input_sentence_1']} Sentence2: {example['input_sentence_2']} Sentence3: {example['input_sentence_3']} Sentence4: {example['input_sentence_4']}"
        sni_instance_input += (
            f"\n(A) {example['sentence_quiz1']} (B) {example['sentence_quiz2']}"
        )

        sni_instance_output = "A" if example["answer_right_ending"] == 1 else "B"

        sni_instances.append(
            {"input": sni_instance_input, "output": sni_instance_output}
        )

    return sni_instances


# In[11]:


dataset = load_xstory_cloze_dataset(lang="en", split="test")


# In[12]:


get_xstorycloze_sni_instances(dataset)


# In[13]:


dataset[0]


# In[15]:


langs = ["en", "sw", "te", "my"]

system_prompt_role = "user"
lang2score = {}
out_dir = "analysis/results/explanations/xstorycloze/{}"
valid_labels = ["A", "B"]
for seed in [42]:
    for include_explanation in [False, True]:
        for lang in langs:
            print(
                f"Running for {lang}, with Explanations: {include_explanation} and seed: {seed}"
            )
            out_dir = f"analysis/results/explanations/xstorycloze/{model}/{lang}/explanations{include_explanation}_system_prompt_role{system_prompt_role}_seed{seed}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            filename = glob.glob(
                f"data/natural-instructions-2.8/tasks/task296_storycloze_correct_end_classification.json"
            )[0]
            with open(filename) as f:
                sni_xstory = json.load(f)

            dataset = load_xstory_cloze_dataset(lang=lang, split="test")
            sni_instances = get_xstorycloze_sni_instances(dataset)
            preds_w_exp = []
            preds_wo_exp = []
            labels = []
            matches = []

            pbar = tqdm(sni_instances)
            for test_example in pbar:
                num_fs = K
                while True:
                    try:
                        prompt = construct_sni_prompt(
                            sni_xstory,
                            test_example,
                            include_explanation=include_explanation,
                            system_prompt_role=system_prompt_role,
                            k=num_fs,
                        )
                        out = gpt3x_completion(
                            prompt,
                            model=model,
                            max_tokens=500 if include_explanation else 10,
                            temperature=TEMPERATURE,
                            timeout=60,
                        )
                        break
                    except openai.error.Timeout:
                        if num_fs >= 0:
                            num_fs -= 1
                            print(
                                f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {(num_fs)}"
                            )
                        else:
                            print(
                                "Exausted Everything! Giving Random Prediction Now :("
                            )
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
                "dataset": "xstory_cloze",
                "lang": lang,
                "include_explanation": include_explanation,
                "system_prompt_role": system_prompt_role,
                "seed": seed,
                "metrics": {"accuracy": acc},
            }
            with open(f"{out_dir}/results.json", "w") as f:
                json.dump(results_dict, f, indent=False)


# In[16]:


preds_w_exp


# In[18]:


preds_wo_exp


# In[ ]:
