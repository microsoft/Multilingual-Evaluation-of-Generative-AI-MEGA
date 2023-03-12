from typing import Union, List, Dict, Tuple, Optional
from promptsource.templates import Template, DatasetTemplates
import pdb
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

def construct_langchain_qa_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    train_prompt_template: str,
    test_prompt_template: str = None,
):
    def preprocess_qa_examples(example):
        return {
            "context": example["context"],
            "question": example["question"],
            "answer": example["answers"]["text"][0],
        }

    if test_prompt_template is None:
        test_prompt_template = train_prompt_template
    example_prompt = PromptTemplate(input_variables=["context", "question", "answer"], template=train_prompt_template)
    if len(train_examples) != 0:
        train_examples = list(map(preprocess_qa_examples, train_examples))
        prompt = FewShotPromptTemplate(
            examples=train_examples,
            example_prompt=example_prompt,
            suffix=test_prompt_template.replace("{answer}", ""),
            input_variables=["context", "question"],
        )
    else:
        prompt = PromptTemplate(input_variables=["context", "question"], 
                                template=test_prompt_template.replace("{answer}", ""))

    return prompt



def construct_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
        train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
        test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns:
        Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    train_prompts = [
        "\n".join(train_prompt_template.apply(train_example))
        for train_example in train_examples
    ]
    test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
    prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    return prompt_input, test_prompt_label


def load_prompt_template(lang: str, prompt_name: str, dataset: str) -> Template:
    """Loads prompt template from promptsource

    Args:
        lang (str): Language specifying the split of xnli dataset for which prompt template is to be loaded
        prompt_name (str): Name of the prompt. Example: GPT-3 style

    Returns:
        Template
    """
    if dataset == "xnli" and lang in set(
        ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "bn"]
    ):
        dataset_prompts = DatasetTemplates(f"Divyanshu/indicxnli/{lang}")
    elif dataset == "xcopa" and lang == "en":
        # For xcopa english data, we need to fetch from COPA in superglue instead
        dataset_prompts = DatasetTemplates("super_glue/copa")
    if dataset == "xnli" and lang in set([]):
        dataset_prompts = DatasetTemplates(f"Divyanshu/indicxnli/{lang}")
    else:
        dataset_prompts = DatasetTemplates(f"{dataset}/{lang}")
    return dataset_prompts[prompt_name]
