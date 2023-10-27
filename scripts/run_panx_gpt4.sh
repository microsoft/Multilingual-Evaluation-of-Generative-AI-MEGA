#!/bin/bash

for lang in en af el eu gu id ka ml nl qu ta tr yo bg he it kk mr pa ro te uk zh
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model "gpt-35-turbo-16k" --max_tokens 100 --chat-prompt --temperature 0 --eval_on_val --num_evals_per_sec 2  -e gpt4v2
done

for lang in ur fa
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model "gpt-35-turbo-16k" --max_tokens 100 --chat-prompt --temperature 0 --num_evals_per_sec 2 --eval_on_val -e gpt4v2 --delimiter ":"
done