#!/bin/bash
# echo "Evaluating for TyDiQA Monolingual"
# for lang in en ar bn fi id ko ru sw te 
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gpt4 -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d tydiqa
# done

# echo "Evaluating for TyDiQA Zero-Shot Cross-Lingual"
# for lang in ar bn fi id ko ru sw te 
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gpt4 -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d tydiqa
# done

# echo "Evaluating for XQuAD"
# for lang in ar el hi ru th tr vi en de es zh ro
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gpt4 -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d xquad
# done


# echo "Evaluating for IndicQA"
# for lang in as bn gu hi kn ml mr or pa ta te
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question+unaswerable" --tgt_prompt_name "answer_given_context_and_question+unaswerable" -k 4 --model "gpt-35-tunro" -e gpt4v2 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --eval_on_val --chat-prompt -d indicqa
# done

echo "Evaluating for IndicQA"
for lang in hi as bn gu kn ml mr or pa ta te
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gptindex -p en -t $lang --pivot_prompt_name "answer_given_context_and_question+unaswerable" --tgt_prompt_name "answer_given_context_and_question+unaswerable" -k 4 --model "text-davinci-003" -e gpt4v2 --temperature 0 --num_evals_per_sec 2 --eval_on_val --chat-prompt -d indicqa
done


# echo "Evaluating for MLQA"
# for lang in en ar de es hi vi zh
# for lang in hi
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gpt-35-tunro" -e gpt4v2 --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --chat-prompt -d mlqa
# done

# echo "Evaluating for MLQA"
# # for lang in ar de es hi vi zh
# for lang in hi
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --chat-prompt -d mlqa --translate-test
# done

# echo "Evaluating for MLQA"
# for lang in ar de es hi vi zh
# do
#     echo "Running for language $lang"
#     python -m mega.eval_qa_gpt4 -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "gptturbo" -e vellm --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --chat-prompt -d mlqa
# done