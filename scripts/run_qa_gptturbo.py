


# In[3]:


import time
from typing import List
import spacy
import openai
import numpy as np
import wandb
from datasets import load_dataset
from mega.data.load_datasets import load_xnli_dataset
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.prompting.prompting_utils import load_prompt_template
from mega.utils.env_utils import load_env
from mega.models.completion_models import get_model_pred, gpt3x_completion
from mega.prompting.prompting_utils import construct_prompt, construct_qa_prompt
from tqdm.notebook import tqdm
from evaluate import load


# In[4]:


# Make sure that {env_name}.env file is present in the envs/ directory
env_name = "melange"
load_env(env_name=env_name)



model = "gpt-35-turbo-deployment"
pivot_lang = "en"
tgt_lang = "ta"
prompt_name = "answer_given_context_and_question"
few_shot_k = 0
dataset = "indicqa"
short_contexts = False
max_tokens = 20


# In[8]:


config = {
    "model" : model,
    "pivot_lang": pivot_lang,
    "tgt_lang": tgt_lang,
    "prompt_name": prompt_name,
    "few_shot_k": few_shot_k,
    "dataset": dataset,
    "short_contexts": short_contexts,
    "max_tokens": max_tokens
}

wandb.init(project="GPT-4-eval", entity="scai-msri", config=config)


# In[9]:


class SpacySentenceTokenizer:
    
    def __init__(self):
        self.nlp = spacy.load('xx_ent_wiki_sm')
        self.nlp.add_pipe("sentencizer")
        
    def __call__(self, text: str) -> List[str]:
        return list(map(lambda span: span.text, self.nlp(text).sents))


# In[10]:


def load_qa_dataset(dataset_name, lang, split, dataset_frac = 1, translate_test = False):
    if dataset_name == "indicqa":
        if split != "train":
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")[split]
        else:
            dataset = load_dataset("squad")[split]
    elif dataset_name == "xquad":
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


# In[11]:


train_dataset = load_qa_dataset(dataset,
                                lang = pivot_lang,
                                split="train")
test_dataset = load_qa_dataset(dataset,
                                lang = tgt_lang,
                                split="validation")


# In[12]:


if short_contexts:
    sent_tokenizer = SpacySentenceTokenizer() 

    train_dataset = train_dataset.map(lambda example: {
        "context": [sent for sent in sent_tokenizer(example["context"]) if example["answers"]["text"][0] in sent][0]
    }, num_proc = 8)


# In[13]:


train_examples = choose_few_shot_examples(
        train_dataset, few_shot_k, selection_criteria="random")


# In[14]:


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


# In[15]:


prompt_template = PROMPTS_DICT[prompt_name]


# In[16]:


# Loading instruction for the task
instruction = INSTRUCTIONS["xquad"]
print(instruction)


# In[17]:


squad_metric = load("squad")


# In[18]:


test_example = test_dataset[132]

prompt, label = construct_qa_prompt(
    train_examples,
    test_example,
    train_prompt_template=prompt_template,
    test_prompt_template=prompt_template,
    chat_prompt=True,
    instruction=instruction
)
prompt


# In[19]:


pred = gpt3x_completion(
    prompt,
    model,
    temperature=0,
    max_tokens=20
)


# In[20]:


print(f"Prediction: {pred}")
print(f"Label: {label}")
prediction = {"prediction_text": pred, "id": test_example["id"]}
reference = {}
reference["answers"] = test_example["answers"]
reference["id"] = test_example["id"]
results = squad_metric.compute(
            predictions=[prediction],
            references=[reference]
        )
print(results)


# In[ ]:


f1_sum = 0
em_sum = 0
avg_em = 0
avg_f1 = 0

run_details = {"num_calls": 0}

pbar = tqdm(enumerate(test_dataset))

for i, test_example in pbar:    
    prompt, label = construct_qa_prompt(
        train_examples,
        test_example,
        train_prompt_template=prompt_template,
        test_prompt_template=prompt_template,
        chat_prompt=True,
        instruction=instruction
    )
    pred = gpt3x_completion(
        prompt,
        model,
        temperature=0,
        run_details=run_details,
        max_tokens=max_tokens
    )
    prediction = {"prediction_text": pred, "id": test_example["id"]}
    reference = {}
    reference["answers"] = test_example["answers"]
    reference["id"] = test_example["id"]
    results = squad_metric.compute(
                predictions=[prediction],
                references=[reference])
    f1_sum += results["f1"]
    em_sum += results["exact_match"]
        
    avg_f1 = f1_sum / (i+1)
    avg_em = em_sum / (i+1)
    
    wandb.log({"f1": avg_f1, "em": avg_em}, step = i+1)
    wandb.log(run_details, step = i+1)
    pbar.set_description(f"em: {avg_em} f1: {avg_f1}")
    time.sleep(1/2)

wandb.finish()

# In[ ]:




