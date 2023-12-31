echo "Zero-shot Evaluation"
echo "Running with Short contexts"
echo "Running with Short contexts"
for k in 4 8
do
    for lang in en ar de el es hi ro ru th tr vi zh
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptindex -p en -t $lang -d xquad --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k --short_contexts
    done
done

echo "Running with Long Contexts"
for k in 4 8
do
    for lang in en ar de el es hi ro ru th tr vi zh
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptindex -p en -t $lang -d xquad --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k
    done
done

