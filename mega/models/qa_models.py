import os
import requests
import time
from typing import Union
from dotenv import load_dotenv
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
import pdb
load_dotenv('env.env')

openai.api_base = "https://gpttesting1.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2022-12-01"  # this may change in the future
HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
BLOOMZ_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"

with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().split("\n")[0]
    
with open("keys/hf_key.txt") as f:
    HF_API_TOKEN = f.read().split("\n")[0]

openai.deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]
openai.embedding_deployment_id = os.environ["EMBEDDING_DEPLOYMENT_ID"]
openai.embedding_deployment_name = os.environ["EMBEDDING_DEPLOYMENT_ID"]


EMBEDDING_LLM = OpenAIEmbeddings(
    document_model_name=openai.embedding_deployment_name,
    query_model_name=openai.embedding_deployment_name,
    openai_api_key=openai.api_key,
)

LLM = AzureOpenAI(
    deployment_name=openai.deployment_name,
    openai_api_key=openai.api_key,
    temperature=0,
    model_kwargs={
        "api_base": openai.api_base,
        "api_type": "azure",
        "api_version": openai.api_version,
    },
)


def answer_question_langchain(
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
        time.sleep(1/5)
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
        k = 1
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

        
        # except openai.error.InvalidRequestError as e:
        #     pdb.set_trace()
        #     response = ""
        #     break

    return response


def answer_question_bloomz(
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(BLOOMZ_API_URL, headers=headers, json=payload)
        return response.json()
    template_format = prompt.example_prompt.template
    examples = prompt.examples
    
    few_shot_ex_prompt = ""
    for example in examples:
        context = context[0] if isinstance(example['context'], list) else example['context']
        template_filled = template_format\
                            .replace('{context}', context)\
                            .replace('{question}', example['question'])\
                            .replace('{answer}', example['answer'])
        few_shot_ex_prompt += f"{template_filled}\n\n"
    
    test_ex_prompt = template_format.replace("{context}", context).replace("{question}", question).replace("{answer}","")
    
    full_prompt = few_shot_ex_prompt + test_ex_prompt
    output = ""
    # prompt = "\n".join([pr]) 
    while True:
        try:
            # signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(full_prompt)
            output = model_output[0]["generated_text"][len(full_prompt):].split("\n")[0]
            output = output.strip()
            # signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if "error" in model_output and "must have less than 1000 tokens." in model_output["error"]:
                raise openai.error.InvalidRequestError(model_output["error"],
                                                       model_output["error_type"])
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            # signal.alarm(0)  # Reset the alarm
            time.sleep(60)
            continue

    return output

def answer_question(
    model : str,
    question: str,
    context: str,
    prompt: Union[PromptTemplate, FewShotPromptTemplate],
    chunk_size: int = 100,
    chunk_overlap: int = 0,
):
    if model == "BLOOMZ":
        return answer_question_bloomz(
            question,
            context,
            prompt,
            chunk_size,
            chunk_overlap
        )
    
    elif model == "DaVinci003":
        return answer_question_langchain(
            question,
            context,
            prompt,
            chunk_size,
            chunk_overlap
        )
    
    else:
        raise NotImplementedError()