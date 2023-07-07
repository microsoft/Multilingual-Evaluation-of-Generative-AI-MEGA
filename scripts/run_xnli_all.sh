#!/bin/bash
# echo "English Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     for lang in en
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gptturbo" -e vellm --chat-prompt --temperature 0
#     done
# done
# echo "Monolingual Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
#     for lang in ar
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --log_wandb
#     done
# done

# echo "Translate-Test Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     for lang in hi ru sw th tr ur vi zh
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --translate-test --model "gptturbo" -e vellm --chat-prompt --temperature 0 --log_wandb
#     done
# done

# echo "Translate-Test Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     for lang in fr
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --translate-test --model "gptturbo" -e vellm --chat-prompt --temperature 0 --log_wandb
#     done
# done

# echo "Zero-shot Evaluation"
# for prompt_name in "GPT-3 style"
# do
#     # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
#     for lang in bg de el es fr hi ru sw th tr ur vi zh
#     do

#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-35-turbo-deployment" -e melange --chat-prompt --temperature 0 --log_wandb

#     done
# done

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for lang in en
    do
        for k in 0 2 4 8 16
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-35-tunro" -e gpt4v3 --chat-prompt --temperature 0 --log_wandb --timeout 30
        done
    done
done