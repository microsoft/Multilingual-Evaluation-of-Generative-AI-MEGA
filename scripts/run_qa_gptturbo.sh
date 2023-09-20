#!/bin/bash

echo "Evaluating for IndicQA"
for lang in hi as bn gu kn ml mr or pa ta te
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gptindex -p en -t $lang --pivot_prompt_name "answer_given_context_and_question+unaswerable" --tgt_prompt_name "answer_given_context_and_question+unaswerable" -k 4 --model "text-davinci-003" -e gpt4v2 --temperature 0 --num_evals_per_sec 2 --eval_on_val --chat-prompt -d indicqa
done
