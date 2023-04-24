#!/bin/bash
# for lang in en es de fr af ar az bg bn
for lang in el es et eu fa fi gu he hi hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --model "gptturbo" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e vellm
done