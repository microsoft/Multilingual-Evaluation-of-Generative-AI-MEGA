import argparse
from mega.models.completion_models import SUPPORTED_MODELS


def parse_args(args: list) -> argparse.Namespace:
    """Parses the arguments provided in the command line

    Args:
        args (list): List of command line arguments to parse

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("Evaluate GPT-x models on XNLI")
    parser.add_argument(
        "-e",
        "--env",
        default="melange",
        choices=["melange", "scai", "vellm", "gpt4", "gpt4v2", "gpt4v3"],
        help="Name of the environment located in envs/",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="xnli",
        # choices=["xnli", "paws-x", "xcopa", "tydiqa", "xquad", "mlqa", "indicqa", "udpos", "panx", "xstory_cloze"],
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
        "-t",
        "--tgt_lang",
        default="en",
        # choices=["en", "hi"],
        type=str,
        help="Language to evaluate on",
    )
    parser.add_argument(
        "--pivot_prompt_name",
        default="GPT-3 style",
        type=str,
        help="Prompt name available in promptsource to use for Pivot",
    )
    parser.add_argument(
        "--tgt_prompt_name",
        default="GPT-3 style",
        type=str,
        help="Prompt name available in promptsource to use for Target",
    )
    parser.add_argument(
        "--same_prompt_name",
        action="store_true",
        help="Whether to use the same prompt type for pivot and target language. Useful for sweeps",
    )
    parser.add_argument(
        "-k", "--few_shot_k", default=4, type=int, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--few_shot_selection",
        default="random",
        choices=["random", "first_k", "random_atleast_one_unanswerable"],
        type=str,
        help="How to select few-shot examples",
    )
    parser.add_argument(
        "--test_frac",
        default=1.0,
        type=float,
        help="Fraction of test data to evaluate on",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random Seed")
    parser.add_argument(
        "--model",
        default="gpt-35-turbo",
        choices=SUPPORTED_MODELS,
        type=str,
        help="GPT-x model to use to evaluate",
    )
    parser.add_argument(
        "--save_dir", default="results", type=str, help="Path to store results"
    )
    parser.add_argument(
        "--translate-test",
        action="store_true",
        help="Whether to use translated test data",
    )
    parser.add_argument(
        "--use-val-to-prompt",
        action="store_true",
        help="Whether to use Validation Data for in-context examples",
    )
    parser.add_argument(
        "--eval_on_val",
        action="store_true",
        help="Whether to use Validation Data for in-evaluation",
    )
    parser.add_argument(
        "--num_evals_per_sec",
        default=2,
        type=int,
        help="Number of evaluations to run per second.",
    )
    parser.add_argument(
        "--parallel_eval",
        dest="parallel_eval",
        action="store_true",
        help="Whether to run parallel evaluation for speedup",
    )
    parser.add_argument(
        "--no-parallel_eval",
        dest="parallel_eval",
        action="store_false",
        help="Whether to run parallel evaluation for speedup",
    )
    parser.set_defaults(parallel_eval=False)
    parser.add_argument(
        "--num_proc",
        default=4,
        type=int,
        help="Number of processes to run parallely for evaluation. Only relevant for parallel_eval.",
    )
    parser.add_argument(
        "--temperature",
        default=0,
        type=float,
        help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,\
            while lower values like 0.2 will make it more focused and deterministic",
    )
    parser.add_argument(
        "--top_p",
        default=1,
        type=float,
        help="An alternative to sampling with temperature, called nucleus sampling,\
            where the model considers the results of the tokens with top_p probability mass.\
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
    )
    parser.add_argument(
        "--max_tokens",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log experiments and results on wandb",
        default=False,
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Whether to not save any results"
    )
    parser.add_argument(
        "--short_contexts",
        action="store_true",
        help="Whether to use short contexts for qa tasks",
    )
    parser.add_argument(
        "--xtreme_dir",
        type=str,
        default="xtreme/download",
        help="Directory containing xtreme datasets",
    )
    parser.add_argument(
        "--copa_dir",
        type=str,
        default="data/copa/",
        help="Directory containing copa datasets",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="_",
        help="Delimiter for token tags for tagging tasks",
    )
    parser.add_argument(
        "--verbalizer",
        type=str,
        default="identity",
        help="Verbalizer type to use. Only applicable for tagging tasks",
    )
    parser.add_argument(
        "--not_one_shot_tag",
        action="store_true",
        help="Whether to not use one-shot tagging. will be slower but more accurate",
    )
    parser.add_argument(
        "--chat-prompt",
        action="store_true",
        help="Whether to use chat type prompts",
    )
    parser.add_argument(
        "--contam_lang",
        default="",
        help="For what languages to check data contamination",
    )

    parser.add_argument(
        "--contam_method",
        default="generate",
        choices=[
            "generate",
            "generate_few_shot",
            "complete",
            "fill_dataset_card",
            "fill_dataset_card_w_example",
        ],
        help="How to check data contamination",
    )

    parser.add_argument(
        "--use_json_format",
        action="store_true",
        help="Whether to use json format for prompting for contamination",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout for each call to the model. 0 means no timeout",
    )
    return parser.parse_args(args)
