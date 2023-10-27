import wandb
from functools import partial
from mega.eval_xnli import main


def sweep():
    prompt_names = [
        "English GPT-3 style",
        "Handcrafted GPT-3 style",
        "English MNLI crowdsource",
        "Handcrafted MNLI crowdsource",
        "English based on the previous passage",
        "Handcrafted based on the previous passage",
    ]
    sweep_config = {
        "method": "grid",
        "name": "xnli_monolingual",
        "parameters": {
            "pivot_lang": {"values": ["hi"]},
            "tgt_lang": {"values": ["hi"]},
            "pivot_prompt_name": {"values": ["English GPT-3 style"]},
            "tgt_prompt_name": {"values": prompt_names},
            "few_shot_k": {"values": [4, 8]},
            "temperature": {"values": [0, 0.2, 0.8, 0.1]},
            "num_proc": {"values": [24]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project="MEGA")
    wandb.agent(
        sweep_id,
        function=partial(main, ["--log_wandb", "--same_prompt_name", "--eval_on_val"]),
        count=48,
    )


if __name__ == "__main__":
    sweep()
