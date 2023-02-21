import os
from tqdm import tqdm
import pdb
import requests, uuid, json
from typing import Union, Optional
import copy
from datasets import Dataset, load_dataset

# Translator setup for bing
endpoint = "https://api.cognitive.microsofttranslator.com/"
with open("keys/bing_translate_key.txt") as f:
    subscription_key = f.read().split("\n")[0]

# Add your location, also known as region. The default is global.
# This is required if using a Cognitive Services resource.
location = "centralindia"
path = "/translate?api-version=3.0"
constructed_url = endpoint + path

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}


def translate_with_bing(text: str, src: str, dest: str) -> str:
    """Uses the bing translator to translate `text` from `src` language to `dest` language

    Args:
        text (str): Text to translate
        src (str): Source language to translate from
        dest (str): Language to translate to

    Returns:
        str: Translated text
    """
    params = {"api-version": "3.0", "from": src, "to": [dest]}
    body = [{"text": text}]

    try:
        request = requests.post(
            constructed_url, params=params, headers=headers, json=body
        )
        response = request.json()
        # pdb.set_trace()
        translation = response[0]["translations"][0]["text"]
    except:
        pdb.set_trace()
        translation = "<MT Failed>"

    return translation


def translate_xnli(
    xnli_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate premise and hypothesis of xnli dataset

    Args:
        xnli_dataset (Dataset): Some split (train, test, val) of XNLI dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate premise
    xnli_dataset = xnli_dataset.map(
        lambda example: {"premise": translate_with_bing(example["premise"], src, dest)},
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate hypothesis
    xnli_dataset = xnli_dataset.map(
        lambda example: {
            "hypothesis": translate_with_bing(example["hypothesis"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        xnli_dataset.save_to_disk(save_path)

    return xnli_dataset

def translate_pawsx(
    pawsx_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate s1 and s2 of pawsx dataset

    Args:
        pawsx_dataset (Dataset): Some split (train, test, val) of pawsx dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate premise
    pawsx_dataset = pawsx_dataset.map(
        lambda example: {"sentence1": translate_with_bing(example["sentence1"], src, dest)},
        num_proc=4,
        load_from_cache_file=False
    )

    # Translate hypothesis
    pawsx_dataset = pawsx_dataset.map(
        lambda example: {
            "sentence2": translate_with_bing(example["sentence2"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False
    )

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pawsx_dataset.save_to_disk(save_path)

    return pawsx_dataset