#!/bin/bash
for lang in fr af ar az bg bn
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --model "gpt-35-turbo-deployment" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e melange
done