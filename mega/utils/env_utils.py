import os
import openai
from dotenv import load_dotenv


def load_env(env_name="melange"):
    load_dotenv(f"envs/{env_name}.env")
    openai.api_base = os.environ["END_POINT"]
    openai.api_type = os.environ["API_TYPE"]
    openai.api_version = os.environ["API_VERSION"]
    openai.api_key = os.environ["API_KEY"]
    openai.deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]
    openai.embedding_deployment_id = os.environ["EMBEDDING_DEPLOYMENT_ID"]
    openai.embedding_deployment_name = os.environ["EMBEDDING_DEPLOYMENT_NAME"]
