#!/bin/bash
# echo "English Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     for lang in en
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-4-32k" -e gpt4 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb
#     done
# done
echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for lang in sw tr ur vi
    do
        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-4-32k" -e gpt4v2 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb
    done
done
