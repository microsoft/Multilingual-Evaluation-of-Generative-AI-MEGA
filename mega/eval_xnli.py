import os
import argparse
import sys
import random
import json
from typing import Union, List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import openai
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates, Template
from mega.utils.translator_utils import translate_xnli
import pdb

openai.api_base = "https://gpttesting1.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2022-12-01"  # this may change in the future

with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().split("\n")[0]


def parse_args(args: list) -> argparse.Namespace:
    """Parses the arguments provided in the command line

    Args:
        args (list): List of command line arguments to parse

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("Evaluate GPT-x models on XNLI")
    parser.add_argument(
        "-p",
        "--pivot_lang",
        required=True,
        choices=["en", "hi"],
        type=str,
        help="Language in which few-shot examples are provided",
    )
    parser.add_argument(
        "-t",
        "--tgt_lang",
        required=True,
        choices=["en", "hi"],
        type=str,
        help="Language to evaluate on",
    )
    parser.add_argument(
        "--pivot_prompt_name",
        required=True,
        type=str,
        help="Prompt name available in promptsource to use for Pivot",
    )
    parser.add_argument(
        "--tgt_prompt_name",
        required=True,
        type=str,
        help="Prompt name available in promptsource to use for Target",
    )
    parser.add_argument(
        "-k", "--few_shot_k", default=4, type=int, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--few_shot_selection",
        default="random",
        choices=["random", "first_k"],
        type=str,
        help="How to select few-shot examples",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random Seed")
    parser.add_argument(
        "--model", default="DaVinci003", type=str, help="GPT-x model to use to evaluate"
    )
    parser.add_argument(
        "--save_dir", default="results", type=str, help="Path to store results"
    )
    parser.add_argument(
        "--translate-test",
        action="store_true",
        help="Whether to use translated test data",
    )
    parser.add_argument(
        "--use-val-to-prompt",
        action="store_true",
        help="Whether to use Validation Data for in-context examples",
    )
    return parser.parse_args(args)


def load_xnli_dataset(lang: str) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    return load_dataset("xnli", lang)


def load_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:

    tt_dir = os.path.join(
        data_dir, "xnli", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_xnli(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_prompt_template(lang: str, prompt_name: str) -> Template:
    """Loads prompt template from promptsource

    Args:
        lang (str): Language specifying the split of xnli dataset for which prompt template is to be loaded
        prompt_name (str): Name of the prompt. Example: GPT-3 style

    Returns:
        Template
    """
    dataset_prompts = DatasetTemplates(f"xnli/{lang}")
    return dataset_prompts[prompt_name]


def choose_few_shot_examples(
    train_dataset: Dataset, few_shot_size: int, selection_criteria: int
) -> List[Dict[str, Union[str, int]]]:
    """Selects few-shot examples from training datasets

    Args:
        train_dataset (Dataset): Training Dataset
        few_shot_size (int): Number of few-shot examples
        selection_criteria (few_shot_selection): How to select few-shot examples. Choices: [random, first_k]

    Returns:
        List[Dict[str, Union[str, int]]]: Selected examples
    """
    example_idxs = []
    if selection_criteria == "first_k":
        example_idxs = list(range(few_shot_size))
    elif selection_criteria == "random":
        example_idxs = (
            np.random.choice(len(train_dataset), size=few_shot_size, replace=False)
            .astype(int)
            .tolist()
        )
    else:
        raise NotImplementedError()

    return [train_dataset[idx] for idx in example_idxs]


def construct_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
        train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
        test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns:
        Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    train_prompts = [
        "\n".join(train_prompt_template.apply(train_example))
        for train_example in train_examples
    ]

    test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)

    prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    return prompt_input, test_prompt_label


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
) -> Dict[str, str]:

    prompt_input, label = construct_prompt(
        train_examples, test_example, train_prompt_template, test_prompt_template
    )

    # Hit the api repeatedly till response is obtained
    while True:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt_input,
                max_tokens=10,
                temperature=1,
                top_p=0.5,
                logprobs=10,
            )
            break
        except openai.error.APIConnectionError:
            continue

    return {"prediction": response["choices"][0]["text"].strip(), "ground_truth": label}


def evaluate_model(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    few_shot_size: int,
    selection_criteria: int,
    save_preds_path: Optional[str] = None,
) -> float:
    """Evaluates the accuracy of the model
    Note: Currently compares the exact match between the generated answer and the verbalized label
    ToDo: Find alternatives to exact match (embeddings?)

    Args:
        train_dataset (Dataset): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        few_shot_size (int): _description_
        selection_criteria (int): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )
    preds = []
    labels = []
    matches = []
    num_matches = 0
    for test_example in tqdm(test_dataset):
        pred_dict = get_model_pred(
            train_examples,
            test_example,
            train_prompt_template,
            test_prompt_template,
            model,
        )
        pred = pred_dict["prediction"]
        label = pred_dict["ground_truth"]
        num_matches += float(pred == label)
        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))

    accuracy = num_matches / len(preds)

    if save_preds_path is not None:
        preds_dir, _ = os.path.split(save_preds_path)
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        results_df = pd.DataFrame(
            {"Label": labels, "Prediction": preds, "Match": matches}
        )
        results_df.to_csv(save_preds_path)

    return accuracy


def main():
    args = parse_args(sys.argv[1:])

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load datasets for pivot and target languages
    train_dataset = load_xnli_dataset(args.pivot_lang)
    if not args.use_val_to_prompt:
        train_dataset = train_dataset["train"]
    else:
        train_dataset = train_dataset["validation"]
    test_dataset = load_xnli_dataset(args.tgt_lang)["test"]
    if args.translate_test:
        test_dataset = load_translate_test(
            args.tgt_lang, args.pivot_lang, test_dataset, data_dir="data"
        )

    # Load prompt templates for train and test datasets
    train_prompt_template = load_prompt_template(
        args.pivot_lang, args.pivot_prompt_name
    )
    test_prompt_template = load_prompt_template(args.tgt_lang, args.tgt_prompt_name)

    train_examples = choose_few_shot_examples(
        train_dataset, args.few_shot_k, args.few_shot_selection
    )

    out_dir = f"{args.save_dir}/xnli/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_FewShotK_{args.few_shot_k}"
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pred_file_path = f"{out_dir}/preds.csv"
    accuracy = evaluate_model(
        train_dataset,
        test_dataset,
        train_prompt_template,
        test_prompt_template,
        args.model,
        args.few_shot_k,
        args.few_shot_selection,
        save_preds_path=pred_file_path,
    )
    print(accuracy)
    # Store results
    results_dict = vars(args)
    results_dict["metrics"] = {"accuracy": accuracy}
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results written in {out_dir}")


if __name__ == "__main__":
    main()
