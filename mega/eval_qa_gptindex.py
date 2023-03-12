import os
import sys
import random
import time
import json
from typing import List
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import openai
from transformers import GPT2Tokenizer
from evaluate import load
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
# from llama_index import LangchainEmbedding
# from llama_index import (
#     GPTSimpleVectorIndex,
#     SimpleDirectoryReader, 
#     LLMPredictor,
#     PromptHelper,
#     StringIterableReader
# )
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
import requests
import numpy as np
from mega.data.load_datasets import load_xnli_dataset, load_xnli_translate_test
from mega.data.data_utils import choose_few_shot_examples
from mega.models.qa_models import answer_question_langchain
from mega.prompting.prompting_utils import construct_langchain_qa_prompt
from mega.utils.parser import parse_args
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

TYDIQA_LANG2CODES = {
    "bengali": "bn",
    "korean" : "ko",
    "swahili" : "sw",
    "english" : "en",
    "indonesian" :"id",
    "arabic" : "ar",
    "finnish" : "fi",
    "telugu" : "te",
    "russian" : "ru"
}

langcodes2lang = {
    "en" : "English",
    "ar" : "Arabic",
    "de" : "German",
    "el" : "Greek",
    "es" : "Spanish",
    "hi" : "Hindi",
    "ro" : "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin"
}


PROMPTS_DICT = {
    "answer_given_context_and_question" : """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    
    "lang_instruct_answer_given_context_and_question" : """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is? Please try to answer in {language} and ensure that the answer appears as it is in the passage.
    A: {answer}""",
    
}

class SpacySentenceTokenizer:
    
    def __init__(self):
        self.nlp = spacy.load('xx_ent_wiki_sm')
        self.nlp.add_pipe("sentencizer")
        
    def __call__(self, text: str) -> List[str]:
        return list(map(lambda span: span.text, self.nlp(text).sents))


def load_qa_dataset(dataset_name, lang, split, dataset_frac = 1, translate_test = False):
    if dataset_name == "xquad":
        if split != "train":
            dataset = load_dataset("xquad", f"xquad.{lang}")[split]
        else:
            dataset = load_dataset("squad")[split]
    elif dataset_name == "tydiqa":
        dataset = load_dataset("tydiqa", 'secondary_task')[split]
        dataset = dataset.map(lambda example: {"lang" : TYDIQA_LANG2CODES[example["id"].split("-")[0]]})
        dataset = dataset.filter(lambda example: example["lang"] == lang)
    elif dataset_name == "mlqa":
        if split == "train":
            print("No Training Data for MLQA, switching to validation!")
            split = "validation"
        if translate_test:
            dataset_name = f"mlqa-translate-test.{lang}"
        else:
            dataset_name = f"mlqa.{lang}.{lang}"
        
        dataset = load_dataset("mlqa", dataset_name)[split]
    
    else:
        raise NotImplementedError()
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def eval_qa(test_dataset, prompt, num_evals_per_sec = 1, smaller_prompts = [], **model_kwargs):
    preds = []
    labels = []
    matches = []
    f1_scores = []
    em_score = 0
    f1_score = 0
    squad_metric = load("squad")
    for test_example in tqdm(test_dataset):
        prompt_to_use = prompt
        for trial in range(0, len(smaller_prompts) + 1):
            try:
                pred = answer_question_langchain(
                    test_example["question"],
                    test_example["context"],
                    prompt=prompt_to_use,
                    chunk_size=model_kwargs.get("chunk_size", 100),
                    chunk_overlap=model_kwargs.get("chunk_overlap", 0),
                ).strip()
                break
            except openai.error.InvalidRequestError as e:
                if trial == len(smaller_prompts):
                    print("Exausted Everything! Giving Empty Prediction Now :(")
                    pred = ""
                    break
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(smaller_prompts) - trial - 1}"
                )
                prompt_to_use = smaller_prompts[trial]
                
        label = test_example["answers"]["text"][0]
        preds.append(pred)
        labels.append(label)
        prediction = {"prediction_text": pred, "id": test_example["id"]}
        reference = {}
        reference["answers"] = test_example["answers"]
        reference["id"] = test_example["id"]
        results = squad_metric.compute(
            predictions=[prediction],
            references=[reference]
        
        )

        matches.append(results["exact_match"]/100)
        em_score += results["exact_match"]
        f1_scores.append(results["f1"])
        f1_score += results["f1"]
        time.sleep(1 / num_evals_per_sec)
    
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches, "F1": f1_scores})

    return {
        "exact_match" : em_score / len(test_dataset),
        "f1" : f1_score / len(test_dataset),
    }, results_df

def main():
    
    args = parse_args(sys.argv[1:])

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    out_dir = f"{args.save_dir}/{args.dataset}_gptindex/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_FewShotK_{args.few_shot_k}"

    if args.short_contexts:
        out_dir = f"{out_dir}_shortContexts"
    
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    train_dataset = load_qa_dataset(args.dataset, lang = args.pivot_lang, split="train" if not args.use_val_to_prompt else "validation")    
    test_dataset = load_qa_dataset(args.dataset,
                                   lang=args.tgt_lang,
                                   split="test" if not args.eval_on_val else "validation",
                                   dataset_frac=args.test_frac,
                                   translate_test=args.translate_test
                                   )

    if args.short_contexts:

        sent_tokenizer = SpacySentenceTokenizer() 
        
        train_dataset = train_dataset.map(lambda example: {
            "context": [sent for sent in sent_tokenizer(example["context"]) if example["answers"]["text"][0] in sent]
        }, num_proc = 24)

    train_examples = choose_few_shot_examples(
        train_dataset, args.few_shot_k, args.few_shot_selection
    )
    train_prompt_template = PROMPTS_DICT[args.pivot_prompt_name]
    test_prompt_template = PROMPTS_DICT[args.tgt_prompt_name]
    if args.pivot_prompt_name == "lang_instruct_answer_given_context_and_question":
        train_prompt_template = train_prompt_template.replace(
            "{language}", langcodes2lang[args.pivot_lang]
        )
    if args.tgt_prompt_name == "lang_instruct_answer_given_context_and_question":
        test_prompt_template = test_prompt_template.replace(
            "{language}", langcodes2lang[args.tgt_lang]
        )
    
    langchain_prompt = construct_langchain_qa_prompt(
        train_examples,
        train_prompt_template=train_prompt_template,
        test_prompt_template=test_prompt_template,
    )
    smaller_prompts = []
    for i in range(1, args.few_shot_k + 1):
        smaller_prompts.append(
            construct_langchain_qa_prompt(
                train_examples[:args.few_shot_k - i],
                train_prompt_template=train_prompt_template,
                test_prompt_template=test_prompt_template,
            )
        )

    metrics, results_df = eval_qa(
        test_dataset,
        langchain_prompt,
        num_evals_per_sec=args.num_evals_per_sec,
        smaller_prompts=smaller_prompts
    )
    
    print(metrics)
    pred_file_path = f"{out_dir}/preds.csv"
    results_df.to_csv(pred_file_path)

    # Store results
    results_dict = vars(args)
    results_dict["metrics"] = metrics
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    # if args.log_wandb:
    #     wandb.log({"accuracy": accuracy})


if __name__ == "__main__":
    main()