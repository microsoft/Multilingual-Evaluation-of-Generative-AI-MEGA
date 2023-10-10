from typing import Dict, List, Union
from collections import Counter
import random
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
    elif selection_criteria == "random_atleast_one_unanswerable":
        unanswerable_idx = None
        for _ in range(100):
            idx = np.random.choice(len(train_dataset))
            if train_dataset[idx]["answers"]["text"] == []:
                unanswerable_idx = idx
                break

        breakpoint()
        if unanswerable_idx is not None:
            example_idxs = [unanswerable_idx]
        else:
            example_idxs = []
        example_idxs += (
            np.random.choice(
                len(train_dataset),
                size=few_shot_size - len(example_idxs),
                replace=False,
            )
            .astype(int)
            .tolist()
        )
        random.shuffle(example_idxs)

    elif selection_criteria == "random-stratified":
        labels = list((train_dataset["label"]))
        label_counts = Counter(labels)
        total = len(train_dataset)
        example_idxs = []
        for label in label_counts:
            label_example_idxs = [
                idx
                for idx, example in enumerate(train_dataset)
                if example["label"] == label
            ]
            sample_size = int(few_shot_size * label_counts[label] / total)
            example_idxs += (
                np.random.choice(label_example_idxs, size=sample_size, replace=False)
                .astype(int)
                .tolist()
            )
        random.shuffle(example_idxs)
    elif selection_criteria == "random-classwise-uniform":
        labels = list(set(train_dataset["label"]))
        example_idxs = []
        sample_size = few_shot_size // len(labels)
        for label in labels:
            label_example_idxs = [
                idx
                for idx, example in enumerate(train_dataset)
                if example["label"] == label
            ]
            example_idxs += (
                np.random.choice(label_example_idxs, size=sample_size, replace=False)
                .astype(int)
                .tolist()
            )
        random.shuffle(example_idxs)
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
            tokens.append(token.strip())
            tags.append(tag.strip())

    return inputs, labels
