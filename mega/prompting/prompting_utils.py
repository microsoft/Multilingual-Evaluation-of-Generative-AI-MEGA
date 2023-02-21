from typing import Union, List, Dict, Tuple, Optional
from promptsource.templates import Template, DatasetTemplates


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
    dataset_prompts = DatasetTemplates(f"{dataset}/{lang}")
    return dataset_prompts[prompt_name]
