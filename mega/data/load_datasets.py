import os
from typing import Union, Optional
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from mega.utils.translator import translate_xnli


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