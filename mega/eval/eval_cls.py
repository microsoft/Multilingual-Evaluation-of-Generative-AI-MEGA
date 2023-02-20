import os
from typing import List, Dict, Union, Tuple, Optional
import time
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from promptsource.templates import Template
from mega.models.completion_models import get_model_pred
from mega.data.data_utils import choose_few_shot_examples


def run_seq_eval(
    train_examples: List[Dict[str, Union[str, int]]],
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    num_evals_per_sec: int = 2,
    **model_params,
) -> Tuple[float, pd.DataFrame]:
    """Runs sequential evaluation. This is slower but would be better when limited API hits are available

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

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
            **model_params,
        )
        pred = pred_dict["prediction"]
        label = pred_dict["ground_truth"]
        num_matches += float(pred == label)
        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))
        time.sleep(1 / num_evals_per_sec)

    accuracy = num_matches / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

    return accuracy, results_df


def run_parallel_eval(
    train_examples: List[Dict[str, Union[str, int]]],
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    num_proc: int = 4,
    **model_params,
) -> Tuple[float, pd.DataFrame]:
    """Runs parallel evaluation.
    This should be substanially fast but should be avoided when limited API hits are available

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.
        num_proc (int): _description_. Defaults to 4.
    Returns:
        _type_: _description_
    """

    results_dataset = test_dataset.map(
        lambda example: get_model_pred(
            train_examples,
            example,
            train_prompt_template,
            test_prompt_template,
            model,
            **model_params,
        ),
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    preds = results_dataset["prediction"]
    labels = results_dataset["ground_truth"]
    matches = [float(pred == label) for (pred, label) in zip(preds, labels)]
    accuracy = sum(matches) / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

    return accuracy, results_df


def evaluate_model(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    few_shot_size: int,
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
    **model_params,
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
        parallel_eval (bool): _description_

    Returns:
        float: _description_
    """

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )

    if parallel_eval:
        num_proc = 4 if num_proc is None else num_proc
        accuracy, results_df = run_parallel_eval(
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model=model,
            num_proc=num_proc,
            **model_params,
        )
    else:
        accuracy, results_df = run_seq_eval(
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model=model,
            **model_params,
        )

    if save_preds_path is not None:
        preds_dir, _ = os.path.split(save_preds_path)
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        results_df.to_csv(save_preds_path)

    return accuracy
