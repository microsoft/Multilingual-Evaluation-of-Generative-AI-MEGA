# Import the necessary modules to run evaluation
from mega.eval.eval_cls import evaluate_model
from mega.data.data_utils import choose_few_shot_examples

# Import datasets and promptsource libraries
from datasets import load_dataset
from promptsource.templates import DatasetTemplates


# Load dataset of your choice
dataset = "paws-x"
src_lang = "en" #Can change the language from en to the language of your choice 
tgt_lang = "en" #Similarly language here can be changed, if it is same as src_lang then monolingual, else zero-shot
train_dataset = load_dataset(dataset, src_lang)["train"] 
test_dataset = load_dataset(dataset, tgt_lang)["test"]
test_dataset = test_dataset.select(list(range(100)))

# Load prompt templates for the dataset
prompt_name = "Meaning" # Name of the prompt created by you on promptsource
train_prompt = DatasetTemplates(f"{dataset}/{src_lang}")[prompt_name]
test_prompt = DatasetTemplates(f"{dataset}/{tgt_lang}")[prompt_name]

# Run evaluation
accuracy = evaluate_model(
        train_dataset,
        test_dataset,
        train_prompt,
        test_prompt,
        model="DaVinci003", #Can change this to BLOOM also
        few_shot_size=4, #Number of few-shot examples
        save_preds_path="results/preds.csv",#Any path where you would like to store predictions,
        temperature=0.1, # Temperature parameter for GPT-3x generations
    )
print(accuracy)