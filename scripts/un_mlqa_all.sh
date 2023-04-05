# echo "Monolingual Evaluation"
# # for k in 8
# # do
# #     # for lang in ar de es hi vi zh
# #     # for lang in de es hi vi zh
# #     for lang in zh
# #     do
# #         echo "Running for lang $lang and k $k"
# #         python -m mega.eval_qa_gptindex -p $lang -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --short_contexts
# #     done
# # done


# # echo "Translate Test Evaluation"
# # for k in 8
# # do
# #     # for lang in ar de es hi vi zh
# #     for lang in ar de es vi zh
# #     do
# #         echo "Running for lang $lang and k $k"
# #         python -m mega.eval_qa_gptindex -p en -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --short_contexts --translate-test
# #     done
# # done

# echo "Zero-Shot Evaluation"
# for k in 8
# do
#     # for lang in ar de es hi vi zh
#     for lang in vi zh
#     do
#         echo "Running for lang $lang and k $k"
#         python -m mega.eval_qa_gptindex -p en -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --short_contexts
#     done
# done

echo "English Evaluation"
for k in 8
do
    for lang in en
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptindex -p $lang -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --short_contexts
    done
done