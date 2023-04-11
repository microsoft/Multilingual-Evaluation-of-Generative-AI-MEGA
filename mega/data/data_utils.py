from typing import Dict, List, Union
import numpy as np
from datasets import Dataset


def choose_few_shot_examples(
    train_dataset: Dataset, few_shot_size: int, selection_criteria: str
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


def read_conll_data(filename: str):

    inputs = []
    labels = []
    with open(filename, "r") as f:
        tokens = []
        tags = []
        for line in f:
            if line == "\n":
                inputs.append(tokens)
                labels.append(tags)
                tokens = []
                tags = []
                continue
            token, tag = line.split("\t")
            tokens.append(token)
            tags.append(tag)

    return inputs, labels
