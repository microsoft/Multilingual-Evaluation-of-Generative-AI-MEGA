import os
import argparse
from typing import Dict, Any, Optional
import openai
import sys
import time
import random
import json
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from mega.data.load_datasets import (
    load_xstory_cloze_dataset,
    load_xstory_cloze_translate_test,
)
from mega.data.data_utils import choose_few_shot_examples
from mega.models.completion_models import model_completion
from mega.prompting.prompting_utils import construct_xstory_prompt
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_env

PROMPT_TEMPLATES = {
    "Answer Given options": """{input_sentence_1} {input_sentence_2} {input_sentence_3} {input_sentence_4}\nWhat is a possible continuation for the story given the following options ?\n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
    "Choose Story Ending": """Read the following story :\n\n{input_sentence_1}\n{input_sentence_2}\n{input_sentence_3}\n{input_sentence_4}\n\nChoose a possible ending for the previous story from the following options:\n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
    "Movie What Happens Next": """Yesterday, I watched a movie. Here''s what happened: {input_sentence_1} {input_sentence_2} {input_sentence_3} {input_sentence_4} What happens next? \n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
    "Story Continuation and Options": """What is a possible continuation for the following story ? \n\n{input_sentence_1}\n{input_sentence_2}\n{input_sentence_3}\n{input_sentence_4}\n\nChoose from the following options:\n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
    "Novel Correct Ending": """I read the following novel: {input_sentence_1} {input_sentence_2} {input_sentence_3} {input_sentence_4} What do you think is the most probable ending? You can choose from the following options:\n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
}

VERBALIZER = {"default": {1: "Option1", 2: "Option2"}}


def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    verbalizer: Dict[Any, str],
    model: str,
    few_shot_size: int,
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    num_evals_per_sec: int = 2,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
    log_wandb: bool = False,
    chat_prompt: bool = False,
    instruction: str = "",
    timeout: int = 0,
    **model_params,
) -> float:
    run_details = {"num_calls": 0}

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )

    valid_labels = [1, 2]

    preds = []
    labels = []
    matches = []
    running_acc = 0
    num_matches = 0
    pbar = tqdm(test_dataset)
    for test_example in pbar:
        train_examples_i = train_examples
        label = verbalizer[test_example["answer_right_ending"]]
        while len(train_examples_i) >= 0:
            prompt, _ = construct_xstory_prompt(
                train_examples_i,
                test_example,
                prompt_template,
                prompt_template,
                verbalizer,
                chat_prompt,
                instruction,
            )
            try:
                pred = model_completion(
                    prompt,
                    model,
                    timeout=timeout,
                    **model_params,
                )
                break
            except (openai.error.InvalidRequestError, openai.error.Timeout):
                if len(train_examples_i) == 0:
                    pred = np.random.choice(valid_labels)
                    print("Exausted Everything! Giving Random Prediction Now :(")
                    break
                train_examples_i = train_examples_i[:-1]
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                )

        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))
        num_matches += float(pred == label)
        running_acc = np.mean(matches)
        pbar.set_description(f"Accuracy: {running_acc}")
        if log_wandb:
            wandb.log({"acuracy": running_acc})
        # time.sleep(1 / num_evals_per_sec)

    accuracy = num_matches / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

    return accuracy, results_df


def main(sys_args):
    args = parse_args(sys_args)
    load_env(env_name=args.env)

    args.dataset = "xstory_cloze"

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project="MEGA",
            entity="msrinlp",
            config=args.__dict__,
        )

        # Need to define these for the sweep
        args.pivot_lang = wandb.config.pivot_lang
        args.tgt_lang = wandb.config.tgt_lang
        args.pivot_prompt_name = wandb.config.pivot_prompt_name
        args.tgt_prompt_name = wandb.config.tgt_prompt_name
        args.few_shot_k = wandb.config.few_shot_k
        args.temperature = wandb.config.temperature
        args.num_proc = wandb.config.num_proc

    # Load datasets for pivot and target languages
    train_dataset = load_xstory_cloze_dataset(
        args.pivot_lang, split="train" if not args.use_val_to_prompt else "validation"
    )
    test_dataset = load_xstory_cloze_dataset(
        args.tgt_lang,
        split="test" if not args.eval_on_val else "validation",
        dataset_frac=args.test_frac,
    )
    if args.translate_test:
        test_dataset = load_xstory_cloze_translate_test(
            args.tgt_lang, args.pivot_lang, test_dataset, data_dir="data"
        )
    # Loading instruction for the task
    instruction = INSTRUCTIONS.get(args.dataset, "")
    print(instruction)

    # Loading prompt template
    prompt_template = PROMPT_TEMPLATES[args.tgt_prompt_name]
    verbalizer = VERBALIZER["default"]

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}wthInstruction"
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"
    if args.eval_on_val:
        out_dir = f"{out_dir}_eval_on_val"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    eval_score, preds_df = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=prompt_template,
        verbalizer=verbalizer,
        model=args.model,
        few_shot_size=args.few_shot_k,
        selection_criteria=args.few_shot_selection,
        num_evals_per_sec=args.num_evals_per_sec,
        parallel_eval=args.parallel_eval,
        num_proc=args.num_proc,
        log_wandb=args.log_wandb,
        chat_prompt=args.chat_prompt,
        instruction=instruction,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    preds_df.to_csv(f"{out_dir}/preds.csv")
    print(eval_score)
    results_dict = vars(args)
    results_dict["metrics"] = {"accuracy": eval_score}
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log({"accuracy": eval_score})


if __name__ == "__main__":
    main(sys.argv[1:])
