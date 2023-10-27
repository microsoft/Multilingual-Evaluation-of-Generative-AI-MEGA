echo "English Evaluation"
for k in 0
do
    for lang in en
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptindex -p $lang -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --model BLOOMZ
    done
done