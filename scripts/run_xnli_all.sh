#!/bin/bash

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for lang in en
    do
        for k in 0 2 4 8 16
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-35-turbo" -e gpt4v3 --chat-prompt --temperature 0 --log_wandb --timeout 30
        done
    done
done