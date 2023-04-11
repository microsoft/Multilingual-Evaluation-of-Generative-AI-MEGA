import requests
import warnings
import signal
import time
import openai
from typing import List, Dict, Union
from mega.prompting.prompting_utils import construct_tagging_prompt


openai.api_base = "https://gpttesting1.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2022-12-01"  # this may change in the future
HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
BLOOMZ_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().split("\n")[0]

with open("keys/hf_key.txt") as f:
    HF_API_TOKEN = f.read().split("\n")[0]

SUPPORTED_MODELS = ["DaVinci003", "BLOOM", "BLOOMZ", "gpt-35-turbo-deployment"]


udpos_verbalizer = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating-conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper-noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating-conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

panx_verbalizer = {
    "B-PER": "begin-person",
    "I-PER": "inside-person",
    "B-ORG": "begin-organization",
    "I-ORG": "inside-organization",
    "B-LOC": "begin-location",
    "I-LOC": "inside-location",
    "O": "non-entity",
}


def gpt3x_tagger(
    prompt: str,
    model: str,
    test_tokens: List[str],
    delimiter: str = "_",
    **model_params,
) -> str:
    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"

        # Hit the api repeatedly till response is obtained
        while True:
            try:
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt_with_token,
                    max_tokens=model_params.get("max_tokens", 20),
                    temperature=model_params.get("temperature", 1),
                    top_p=model_params.get("top_p", 1),
                )
                break
            except (
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.APIError,
            ):
                continue

            except TypeError:
                warnings.warn(
                    "Couldn't generate response, returning empty string as response"
                )
                return ""

        return response["choices"][0]["text"].strip().split()[0]

    if model == "gpt-35-turbo-deployment":
        openai.api_version = "2023-03-15-preview"
    else:
        openai.api_version = "2022-12-01"

    prompt_with_decodings = prompt
    predicted_tags = []
    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tags)

    return predicted_tags


def bloom_tagger(
    prompt: str,
    model: str,
    test_tokens: List[str],
    delimiter: str = "_",
    **model_params,
) -> str:

    assert model in ["BLOOM", "BLOOMZ"]

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def query(payload):
        if model == "bloom":
            payload = payload
            url = HF_API_URL
        else:
            payload = {"inputs": payload}
            url = BLOOMZ_API_URL
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"

        # Hit the api repeatedly till response is obtained
        output = ""
        while True:
            try:
                model_output = query(prompt_with_token)
                output = model_output[0]["generated_text"][
                    len(prompt_with_token) :
                ].split()[0]
                output = output.strip()
                break
            except Exception:
                if (
                    "error" in model_output
                    and "must have less than 1000 tokens." in model_output["error"]
                ):
                    raise openai.error.InvalidRequestError(
                        model_output["error"], model_output["error_type"]
                    )
                print("Exceeded Limit! Sleeping for a minute, will try again!")
                time.sleep(60)
                continue

        return output

    prompt_with_decodings = prompt
    predicted_tags = []
    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tags)

    return predicted_tags


def model_tagger(
    prompt: str,
    model: str,
    test_tokens: List[str],
    delimiter: str = "_",
    **model_params,
) -> str:

    if model in ["DaVinci003", "gpt-35-turbo-deployment"]:
        return gpt3x_tagger(
            prompt,
            model,
            test_tokens,
            delimiter,
            **model_params,
        )
    elif model in ["BLOOM", "BLOOMZ"]:
        return bloom_tagger(
            prompt,
            model,
            test_tokens,
            delimiter,
            **model_params,
        )


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    prompt_template: str,
    verbalizer: Dict[str, str],
    model: str,
    delimiter: str = "_",
    **model_params,
):

    reverse_verbalizer = {value: key for key, value in verbalizer.items()}

    prompt_input, label = construct_tagging_prompt(
        train_examples, test_example, prompt_template, verbalizer
    )
    model_prediction = model_tagger(
        prompt_input,
        model,
        test_tokens=test_example["tokens"],
        delimiter=delimiter,
        **model_params,
    )
    model_prediction_tags = [
        prediction.split(delimiter)[-1] for prediction in model_prediction
    ]
    model_prediction_tags = [
        reverse_verbalizer.get(prediction_tag, prediction_tag)
        for prediction_tag in model_prediction_tags
    ]

    return {"prediction": model_prediction_tags, "ground_truth": label}
