#!/bin/bash
echo "Monolingual Evaluation"
for prompt_name in "Handcrafted GPT-3 style" "English GPT-3 style" "Handcrafted MNLI crowdsource" "English MNLI crowdsource" "Handcrafted based on the previous passage" "English based on the previous passage"
do
    for k in 8 4
    do
        echo "Running for ${prompt_name} and $k few-shot examples"
        python -m mega.eval_xnli -p hi -t hi --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" --num_proc 24 -k $k
    done
done

echo "Translate-Test Evaluation"
for prompt_name in "GPT-3 style" "MNLI crowdsource" "based on the previous passage"
do
    for k in 8 16
    do
        echo "Running for ${prompt_name} and $k few-shot examples"
        python -m mega.eval_xnli -p en -t hi --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "English ${prompt_name}" --num_proc 24 -k $k --translate-test
    done
done

echo "Zero-shot Evaluation"
for prompt_name in "GPT-3 style" "MNLI crowdsource" "based on the previous passage"
do
    for k in 8 4
    do

        echo "Running for ${prompt_name} and $k few-shot examples with English Prompts"
        python -m mega.eval_xnli -p en -t hi --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "English ${prompt_name}" --num_proc 24 -k $k
        echo "Running for ${prompt_name} and $k few-shot examples with Handcrafted Prompts"
        python -m mega.eval_xnli -p en -t hi --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "Handcrafted ${prompt_name}" --num_proc 24 -k $k
    done
done