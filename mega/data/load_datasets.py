import os
import xml.etree.ElementTree as ET
import warnings
from typing import Union, Optional
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from mega.utils.translator import translate_xnli, translate_pawsx, translate_xstory_cloze
from mega.data.data_utils import read_conll_data


def load_xnli_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    if lang in set(
        ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "bn"]
    ):  ##PJ:To add except hindi
        dataset = load_dataset("Divyanshu/indicxnli", lang)[split]
    else:
        dataset = load_dataset("xnli", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xnli_translate_test(
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


def load_pawsx_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which paws-x dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    dataset = load_dataset("paws-x", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_pawsx_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "paws-x", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_pawsx(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_xstory_cloze_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which paws-x dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    if split == "validation":
        split = "train"
    elif split == "test":
        split = "eval"
    dataset = load_dataset("juletxara/xstory_cloze", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xstory_cloze_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:

    tt_dir = os.path.join(
        data_dir, "xstory_cloze", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_xstory_cloze(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def parse_copa_dataset(path, split = "test"):

    tree = ET.parse(f"{path}/copa-{split}.xml")
    root = tree.getroot()
    items = root.findall('item')
    
    dataset = []
    
    for item in items:
        dataset.append(
            {
                "idx": item.get("id"),
                "question": item.get("asks-for"),
                "label": int(item.get("most-plausible-alternative")) - 1,
                "premise": item.find("p").text,
                "choice1": item.find("a1").text,
                "choice2": item.find("a2").text,
            }
        )
    
    return Dataset.from_list(dataset)
            
    

def load_xcopa_dataset(
    lang: str, split: str, 
    dataset_frac: float = 1.0, copa_dir="data/copa/"
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test or validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """

    if lang != "en" and split == "train":
        warnings.warn(
            "No Training Split for Non-English languages in XCOPA. Using Validation split!"
        )
        split = "validation"
    if lang == "en":
        if split in ["train", "validation"]:
            # For english fetch data from COPA in SuperGLUE
            dataset = load_dataset("super_glue", "copa")[split]
        else:
            dataset = parse_copa_dataset(copa_dir, split = "test")
    else:
        dataset = load_dataset("xcopa", lang)[split]

    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_tagging_dataset(
    dataset: str,
    lang: str,
    split: str,
    max_examples: int = -1,
    dataset_frac: float = 1.0,
    xtreme_dir: str = "xtreme/download",
    delimiter: str = "_",
) -> Union[Dataset, DatasetDict]:
    
    split = "dev" if split == "validation" else split

    filename = f"{xtreme_dir}/{dataset}/{split}-{lang}.tsv"
    inputs, labels = read_conll_data(filename)

    dataset = Dataset.from_dict({"tokens": inputs, "tags": labels})
    dataset = dataset.map(
        lambda example: {
            "tagged_tokens": [f"{token}{delimiter}{tag}"
            for token, tag in zip(example["tokens"], example["tags"])]
        }
    )
    N = len(dataset)
    if max_examples == -1:
        selector = np.arange(int(N * dataset_frac))
    else:
        selector = np.arange(min(N, max_examples))
    return dataset.select(selector)
