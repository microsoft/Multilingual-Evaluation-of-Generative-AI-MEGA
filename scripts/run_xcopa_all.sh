#!/bin/bash
echo "Monolingual Evaluation"
for prompt_name in "Bing-Translated plausible_alternatives_discrete" "plausible_alternatives_discrete"
do
    # for lang in en et id it sw ta th tr ht qu
    for lang in en
    do
        if [[ $lang == "en" || $lang == "qu" || $lang == "ht" ]] && [[ ${prompt_name} == "Bing-Translated plausible_alternatives_discrete" ]]; then
            continue
        fi
        if [[ $lang == "ta" ]]
        then
            k=4
        else
            k=8
        fi  
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --test_frac 0.01
    done
done

# echo "Translate-Test Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in translation-et translation-id translation-it translation-sw translation-ta translation-th translation-tr translation-ht translation-qu
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xcopa -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8
#     done
# done

# echo "Zero-shot Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in et id it sw ta th tr ht qu
#     do

#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xcopa -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8

#     done
# done