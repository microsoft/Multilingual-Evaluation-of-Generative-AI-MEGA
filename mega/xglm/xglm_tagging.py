import os
import sys
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import XGLMTokenizer, XGLMForCausalLM
from seqeval.metrics import f1_score
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.prompting_utils import construct_tagging_prompt
from mega.utils.parser import parse_args
from mega.data.load_datasets import load_tagging_dataset
from mega.utils.env_utils import load_openai_env_variables
import openai

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

PROMPTS_DICT = {
    "structure_prompting": """C: {context}\nT: {tagged}""",
    "structure_prompting_chat": """{context}\n{tagged}""",
    "structure_prompting_chat_wth_instruct": """Tag the following sentence: "{context}"\n{tagged}""",
}


def get_xglm_pred(
    model,
    tokenizer,
    train_examples,
    test_example,
    prompt_template,
    label_list,
    delimiter="_",
):
    prompt_input, label = construct_tagging_prompt(
        train_examples,
        test_example,
        prompt_template,
        verbalizer={},
        delimiter=delimiter,
        chat_prompt=False,
    )
    test_tokens = [token for token in test_example["tokens"]]

    def predict_tag(prompt, token):
        # Choose all labels as possible alternatives
        alternatives = label_list

        # Concatenate each {token}{delimiter}{alternative} to the prompt to create a batch
        batch = [prompt + f"{token}{delimiter}{alt}" for alt in alternatives]

        # Tokenize the batch and create a tensor
        tokenized_out = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = tokenized_out["input_ids"].to(model.device)
        output_ids = input_ids[:, 1:].contiguous().to(model.device)
        attn_mask = tokenized_out["attention_mask"].to(model.device)

        # Get the logits from the model and compute total log probabilities for each sequence in the batch
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits
        logprobs = torch.gather(
            F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2)
        )
        log_probs_masked = logprobs * attn_mask[:, 1:].unsqueeze(-1)
        total_log_probs = log_probs_masked.sum(dim=1)

        # Chose the alternative with maximum total_log_probs
        pred_idx = total_log_probs.argmax(dim=0).squeeze().item()
        pred = alternatives[pred_idx]

        return pred

    prompt_with_decodings = prompt_input
    predicted_tags = []
    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tag)
    return {"prediction": predicted_tags, "ground_truth": label}


def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    model: XGLMForCausalLM,
    tokenizer: XGLMTokenizer,
    few_shot_size: int,
    delimiter: str = "_",
    selection_criteria: str = "random",
    log_wandb: bool = False,
) -> float:
    run_details = {"num_calls": 0}

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )

    valid_labels = set()
    for example in train_examples:
        valid_labels.update(example["tags"])
    valid_labels = list(valid_labels)

    preds = []
    labels = []
    f1_scores = []
    pbar = tqdm(test_dataset)
    for test_example in pbar:
        train_examples_i = train_examples

        while len(train_examples_i) >= 1:
            try:
                pred_dict = get_xglm_pred(
                    model,
                    tokenizer,
                    train_examples_i,
                    test_example,
                    prompt_template,
                    label_list=valid_labels,
                    delimiter=delimiter,
                )
                break
            except (openai.error.InvalidRequestError, openai.error.Timeout):
                if len(train_examples_i) == 0:
                    pred_dict = {
                        "prediction": np.random.choice(
                            valid_labels, len(test_example["tags"]), replace=True
                        ).tolist(),
                        "ground_truth": test_example["tags"],
                    }
                    print("Exausted Everything! Giving Random Prediction Now :(")
                    break
                train_examples_i = train_examples_i[:-1]
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                )

        pred_dict["prediction"] = [
            pred if pred != "" else np.random.choice(valid_labels)
            for pred in pred_dict["prediction"]
        ]
        preds.append(pred_dict["prediction"])
        labels.append(pred_dict["ground_truth"])
        try:
            f1_scores.append(f1_score(preds, labels))
        except IndexError:
            breakpoint()
        running_f1 = f1_scores[-1]
        pbar.set_description(f"F1-Score: {running_f1}")
        if log_wandb:
            wandb.log({"f1": running_f1})
        # time.sleep(1 / num_evals_per_sec)

    eval_score = f1_score(labels, preds)
    results_df = pd.DataFrame(
        {"Label": labels, "Prediction": preds, "F1-Score": f1_scores}
    )

    return eval_score, results_df


def main(sys_args):
    args = parse_args(sys_args)

    # Override the model name with xglm
    args.model = "xglm"

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project="MEGA",
            entity="msrinlp",
            config=args.__dict__,
        )

    # Load dataset for pivot and target languages
    train_dataset = load_tagging_dataset(
        args.dataset,
        args.pivot_lang,
        split="train" if not args.use_val_to_prompt else "validation",
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter,
    )
    test_dataset = load_tagging_dataset(
        args.dataset,
        args.tgt_lang,
        split="test" if not args.eval_on_val else "validation",
        max_examples=1000,  # args.max_examples,
        dataset_frac=args.test_frac,
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter,
    )

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}"

    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize XGLM model
    xglm_model_name = "facebook/xglm-7.5B"
    tokenizer = XGLMTokenizer.from_pretrained(xglm_model_name)
    model = XGLMForCausalLM.from_pretrained(xglm_model_name, load_in_8bit=True)
    model.eval()

    eval_score, results_df = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=PROMPTS_DICT[args.tgt_prompt_name],
        model=model,
        tokenizer=tokenizer,
        few_shot_size=args.few_shot_k,
        delimiter=args.delimiter,
        selection_criteria=args.few_shot_selection,
        log_wandb=args.log_wandb,
    )
    print(eval_score)


if __name__ == "__main__":
    main(sys.argv[1:])
