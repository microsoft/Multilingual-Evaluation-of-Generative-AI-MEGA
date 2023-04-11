import requests
import warnings
import signal
import time
import openai
from typing import List, Dict, Union
from promptsource.templates import Template
from mega.prompting.prompting_utils import construct_prompt
import pdb

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

# Register an handler for the timeout
# def handler(signum, frame):
#     raise Exception("API Response Stuck!")

# signal.signal(signal.SIGALRM, handler)


def gpt3x_completion(prompt: str, model: str, **model_params) -> str:

    """Runs the prompt over the GPT3.x model for text completion

    Args:
        - prompt (str) : Prompt String to be completed by the model
        - model (str) : GPT-3x model to use

    Returns:
        str: generated string
    """

    if model == "gpt-35-turbo-deployment":
        openai.api_version = "2023-03-15-preview"
    else:
        openai.api_version = "2022-12-01"

    # Hit the api repeatedly till response is obtained
    while True:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=model_params.get("max_tokens", 20),
                temperature=model_params.get("temperature", 1),
                top_p=model_params.get("top_p", 1),
            )
            break
        except (
            openai.error.APIConnectionError,
            openai.error.RateLimitError,
            openai.error.APIError,
        ) as e:
            continue
        except TypeError:
            warnings.warn(
                "Couldn't generate response, returning empty string as response"
            )
            return ""

    return response["choices"][0]["text"].strip().split("\n")[0]


def bloom_completion(prompt: str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def query(payload):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        return response.json()

    output = ""
    while True:
        try:
            signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if (
                "error_" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.error.InvalidRequestError
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            signal.alarm(0)  # Reset the alarm
            time.sleep(60)
            continue

    return output


def bloomz_completion(prompt: str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(BLOOMZ_API_URL, headers=headers, json=payload)
        return response.json()

    output = ""
    while True:
        try:
            # signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            output = output.strip()
            # signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if (
                "error" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.error.InvalidRequestError(
                    model_output["error"], model_output["error_type"]
                )
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            # signal.alarm(0)  # Reset the alarm
            time.sleep(60)
            continue

    return output


def model_completion(prompt: str, model: str, **model_params) -> str:

    """Runs the prompt over one of the `SUPPORTED_MODELS` for text completion

    Args:
        - prompt (str) : Prompt String to be completed by the model
        - model (str) : Model to use

    Returns:
        str: generated string
    """

    if model in ["DaVinci003", "gpt-35-turbo-deployment"]:
        return gpt3x_completion(prompt, model, **model_params)

    if model == "BLOOM":
        return bloom_completion(prompt, **model_params)

    if model == "BLOOMZ":
        return bloomz_completion(prompt, **model_params)


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    **model_params,
) -> Dict[str, str]:
    """_summary_

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_example (Dict[str, Union[str, int]]): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_

    Returns:
        Dict[str, str]: _description_
    """

    prompt_input, label = construct_prompt(
        train_examples, test_example, train_prompt_template, test_prompt_template
    )

    model_prediction = model_completion(prompt_input, model, **model_params)
    return {"prediction": model_prediction, "ground_truth": label}
