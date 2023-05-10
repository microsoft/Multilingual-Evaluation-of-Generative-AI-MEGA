#!/bin/bash
# echo "Evaluating for TyDiQA"
# for lang in ar bn fi id ru sw te
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gpt4 -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gpt-4-32k" -e gpt4v2 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d tydiqa
# done

echo "Evaluating for XQuAD"
for lang in th tr vi ro
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gpt4 -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gpt-4-32k" -e gpt4v2 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d xquad
done