#!/bin/bash
# for lang in id ko nl ro te ur zh ar el et fi hi it lt pl ru tr vi bg eu fr hu ja mr pt ta uk wo
# do
#     python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --model "gpt-35-tunro" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e gpt4v2
# done

# for lang in af de es he id ko nl ro te zh ar el et fi hi it lt pl ru tr vi bg eu fr hu ja mr pt ta uk wo
# for lang in ro te zh ar el et fi hi it lt pl ru tr vi bg eu fr hu ja mr pt ta uk wo
for lang in en kk
do
    python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --model "gpt-4-32k" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e gpt4v2
done

# for lang in ur fa
# do
#     python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --model "gpt-4-32k" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e "gpt4v2" --delimiter ":"
# done


# for lang in th tl yo
# do
#     python -m mega.eval_tag -d udpos -p en -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model "gpt-4-32k" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb -e "gpt4v2"
# done