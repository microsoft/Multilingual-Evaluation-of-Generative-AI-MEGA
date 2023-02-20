#!/bin/bash
# echo "Monolingual Evaluation"
# for prompt_name in "Bing-Translated based on the previous passage" "based on the previous passage"
# do
#     for lang in ar bg de el es fr hi ru sw th tr ur vi zh
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8
#     done
# done

echo "Translate-Test Evaluation"
for prompt_name in "based on the previous passage"
do
    for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    do
        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --translate-test
    done
done

echo "Zero-shot Evaluation"
for prompt_name in "based on the previous passage"
do
    for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    do

        echo "Running for language $lang and prompt ${prompt_name}"
        python -m mega.eval_xnli -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8

    done
done