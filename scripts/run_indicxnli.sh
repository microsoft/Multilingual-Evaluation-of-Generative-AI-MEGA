#!/bin/bash
echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for lang in as bn gu hi kn ml mr or pa ta te
    do
        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-35-turbo-deployment" -e melange --chat-prompt --temperature 0 --log_wandb
    done
done

echo "Translate-Test Evaluation"
for prompt_name in "GPT-3 style"
do
    for lang in as bn gu hi kn ml mr or pa ta te
    do
        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --translate-test --model "gpt-35-turbo-deployment" -e melange --chat-prompt --temperature 0 --log_wandb
    done
done

echo "Zero-shot Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for lang in as bn gu hi kn ml mr or pa ta te
    do

        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-35-turbo-deployment" -e melange --chat-prompt --temperature 0 --log_wandb

    done
done