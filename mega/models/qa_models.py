import time
import warnings
from typing import List, Union

import openai
import requests
from langchain import VectorDBQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

from mega.models.completion_models import gpt3x_completion
from mega.utils.env_utils import (BLOOMZ_API_URL, HF_API_KEY,
                                  load_openai_env_variables)

load_openai_env_variables()


EMBEDDING_LLM = OpenAIEmbeddings(
    openai_api_key=openai.api_key,
)

LLM = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    openai_api_key=openai.api_key,
    temperature=0,
    model_kwargs={
        "api_base": openai.api_base,
        "api_type": "azure",
        "api_version": openai.api_version,
    },
)

CHAT_LLM = AzureChatOpenAI(
    openai_api_base=openai.api_base,
    openai_api_version="2023-03-15-preview",
    deployment_name="gpt-35-turbo",
    openai_api_key=openai.api_key,
    openai_api_type="azure",
    temperature=0,
)


def answer_question_gpt(
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_text(context)

    embedding = EMBEDDING_LLM
    docsearch = Chroma.from_texts([texts[0]], embedding, metadatas=[{}])
    for text in texts[1:]:
        time.sleep(1 / 5)
        while True:
            try:
                docsearch.add_texts([text], metadatas=[{}])
                break
            except openai.error.APIConnectionError:
                continue
    qa = VectorDBQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        vectorstore=docsearch,
        chain_type_kwargs={"prompt": prompt},
        k=1,
    )

    while True:
        try:
            response = qa.run(question)
            break
        except openai.error.APIConnectionError:
            continue
        except TypeError:
            response = ""
            break

    return response


def answer_question_chatgpt(
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    openai.api_version = "2023-03-15-preview"
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_text(context)
    embedding = EMBEDDING_LLM
    docsearch = Chroma.from_texts([texts[0]], embedding, metadatas=[{}])
    for text in texts[1:]:
        time.sleep(1 / 5)
        while True:
            try:
                docsearch.add_texts([text], metadatas=[{}])
                break
            except (openai.error.APIConnectionError, openai.error.APIError):
                continue

    qa = VectorDBQA.from_chain_type(
        llm=CHAT_LLM,
        chain_type="stuff",
        vectorstore=docsearch,
        chain_type_kwargs={"prompt": prompt},
        k=1,
    )

    while True:
        try:
            response = qa.run(question)
            break
        except openai.error.APIConnectionError:
            continue
        except TypeError:
            response = ""
            break
        except KeyError:
            warnings.warn(
                "ToDo: Some KeyError, yet to figrue out the root to this response. Report to t-sahuja if you see multiple instances of this"
            )
            return ""
    return response


def answer_question_gpt4(
    question: str, context: str, prompt: Union[List, str], model: str, **model_params
):
    return gpt3x_completion(prompt, model, **model_params)


def answer_question_bloomz(
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(BLOOMZ_API_URL, headers=headers, json=payload)
        return response.json()

    if isinstance(prompt, PromptTemplate):
        template_format = prompt.template
        examples = []
    else:
        template_format = prompt.example_prompt.template
        examples = prompt.examples

    few_shot_ex_prompt = ""
    for example in examples:
        context = (
            context[0] if isinstance(example["context"], list) else example["context"]
        )
        template_filled = (
            template_format.replace("{context}", context)
            .replace("{question}", example["question"])
            .replace("{answer}", example["answer"])
        )
        few_shot_ex_prompt += f"{template_filled}\n\n"

    test_ex_prompt = (
        template_format.replace("{context}", context)
        .replace("{question}", question)
        .replace("{answer}", "")
    )

    full_prompt = few_shot_ex_prompt + test_ex_prompt
    output = ""
    # prompt = "\n".join([pr])
    while True:
        try:
            # signal.alarm(60)  # Wait for a minute for the response to come
            breakpoint()
            model_output = query(full_prompt)
            output = model_output[0]["generated_text"][len(full_prompt) :].split("\n")[
                0
            ]
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


def answer_question(
    model: str,
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate, List, str],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    if model == "BLOOMZ":
        return answer_question_bloomz(
            question, context, prompt, chunk_size, chunk_overlap
        )
    elif model in ["gpt-35-turbo"]:
        return answer_question_chatgpt(
            question, context, prompt, chunk_size, chunk_overlap
        )
    elif model in ["gpt-4", "gpt-4-32k"]:
        return answer_question_gpt4(question, context, prompt, model)

    else:
        raise NotImplementedError()
