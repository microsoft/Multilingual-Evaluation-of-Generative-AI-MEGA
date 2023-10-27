echo "Monolingual Evaluation"
for k in 8
do
    for lang in en ar bn fi id sw ko ru te
    # for lang in te bn
    do
        if [[ $lang == "te"  || $lang == "bn" ]];
        then
            k=4
        else
            k=8
        fi
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptturbo -p $lang -t $lang -d tydiqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k --short_contexts
    done
done

# echo "Zero-Shot Evaluation"
# for k in 8
# do
#     for lang in ar bn fi id sw ko ru te
#     do
#         echo "Running for lang $lang and k $k"
#         python -m mega.eval_qa_gptindex -p en -t $lang -d tydiqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k --short_contexts
#     done
# done