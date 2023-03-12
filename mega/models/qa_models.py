import os
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

with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().split("\n")[0]

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