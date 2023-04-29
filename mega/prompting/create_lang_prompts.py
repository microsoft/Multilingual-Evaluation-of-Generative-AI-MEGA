import argparse
from typing import Optional
from word2word import Word2word
from promptsource.templates import Template, DatasetTemplates
from mega.utils.translator import translate_with_bing
from mega.prompting.prompting_utils import load_prompt_template
from mega.models.completion_models import SUPPORTED_MODELS
import pdb

dataset2langs = {"xnli": "ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh"}


def translate_jinja(jinja: str, tgt_lang: str, pivot_lang: str) -> str:

    translated_jinja = jinja
    translated_jinja = translated_jinja.replace(
        "{{", "{{V_"
    )  # A hack to avoid translation of placeholders
    translated_jinja = translated_jinja.replace(
        "[label", "[V_label"
    )  # A hack to avoid translation of placeholders
    translated_jinja = translate_with_bing(
        translated_jinja, src=pivot_lang, dest=tgt_lang
    )
    translated_jinja = translated_jinja.replace("V_", "")
    return translated_jinja


def translate_answer_choices(
    answer_choices: Optional[str], tgt_lang: str, pivot_lang: str
) -> Optional[str]:
    if answer_choices is None:
        return answer_choices

    choices_list = answer_choices.split("|||")
    translated_choices_list = [
        translate_with_bing(choice, pivot_lang, tgt_lang) for choice in choices_list
    ]

    return "|||".join(translated_choices_list)


def human_intervention(
    og_string: str, translated_string: str, string_type: str = "Template"
) -> str:
    print(f"Original {string_type}:\n{og_string}")
    print(f"Translated {string_type}:\n{translated_string}")
    need_correction = input(f"Does the translated {string_type} need correction? Y/N ")
    if need_correction.lower() == "n":
        return translated_string

    corrected_string = input(
        f"Please provide the corrected {string_type}. If not possible, press X.\n"
    )
    if corrected_string.lower() == "x":
        raise ValueError(
            "Can't correct the incorrect Translation. Raise the issue on the repo!"
        )

    return corrected_string


def translate_prompt(prompt_template: Template, tgt_lang: str, pivot_lang: str):

    translated_template_name = f"Bing-Translated {prompt_template.name}"

    translated_jinja = translate_jinja(prompt_template.jinja, tgt_lang, pivot_lang)
    corrected_jinja = human_intervention(
        prompt_template.jinja, translated_jinja, string_type="Template"
    )

    translated_answer_choices = translate_answer_choices(
        prompt_template.answer_choices, tgt_lang, pivot_lang
    )
    corrected_answer_choices = human_intervention(
        prompt_template.answer_choices, translated_answer_choices, string_type="Answers"
    )

    return Template(
        name=translated_template_name,
        jinja=corrected_jinja,
        reference=f"Translated to {tgt_lang} version  of: {prompt_template.reference}",
        metadata=prompt_template.metadata,
        answer_choices=corrected_answer_choices,
    )


def add_prompt_to_dataset(
    tgt_prompt_dataset: DatasetTemplates,
    src_prompt_template: Template,
    tgt_lang: str,
    pivot_lang: str,
    translate=True,
):

    if translate:
        tgt_prompt_template = translate_prompt(
            src_prompt_template, tgt_lang, pivot_lang
        )
    else:
        tgt_prompt_template = src_prompt_template

    tgt_prompt_dataset.add_template(tgt_prompt_template)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="xnli",
        choices=["xnli", "paws-x", "xcopa", "Divyanshu/indicxnli"],
        type=str,
        help="(HF) Dataset to use",
    )
    parser.add_argument(
        "-p",
        "--pivot_lang",
        default="en",
        # choices=["en", "hi"],
        type=str,
        help="Language in which few-shot examples are provided",
    )
    parser.add_argument(
        "--tgt_langs",
        default="es,hi",
        type=str,
        help="Languages to translate to",
    )
    parser.add_argument(
        "--pivot_prompt_name",
        default="GPT-3 style",
        type=str,
        help="Prompt name available in promptsource to use for Pivot",
    )
    parser.add_argument(
        "--model",
        default="DaVinci003",
        choices=SUPPORTED_MODELS,
        type=str,
        help="GPT-x model to use to evaluate",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Whether to not run translation for generating prompts",
    )
    args = parser.parse_args()
    prompt_template = load_prompt_template(
        args.pivot_lang, args.pivot_prompt_name, dataset=args.dataset
    )
    tgt_langs = args.tgt_langs.split(
        ","
    )  # Can provide multiple languages here as comma seperated values
    for tgt_lang in tgt_langs:
        tgt_prompt_dataset = DatasetTemplates(f"{args.dataset}/{tgt_lang}")
        if not args.no_translate:
            print(f"Creating Translated Prompts for {tgt_lang}")
            add_prompt_to_dataset(
                tgt_prompt_dataset,
                prompt_template,
                tgt_lang,
                args.pivot_lang,
                translate=True,
            )
        print(f"Creating Identity Prompts for {tgt_lang}")
        add_prompt_to_dataset(
            tgt_prompt_dataset,
            prompt_template,
            tgt_lang,
            args.pivot_lang,
            translate=False,
        )

    # trannslated_prompt = translate_prompt(prompt_template, "es", "en")
    # pdb.set_trace()


if __name__ == "__main__":
    main()
