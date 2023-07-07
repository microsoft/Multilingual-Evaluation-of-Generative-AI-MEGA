#!/bin/bash
# echo "English Evaluation for DV003"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in en
#     do
#         k=8
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "DaVinci003" -e melange -d xcopa
#     done
# done

# echo "Starting Turbo  Evaluation"
# echo "Monolingual Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in en et id it sw ta th tr ht qu
#     do
#         k=8
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-35-turbo-deployment" -e melange --chat-prompt -d xcopa
#     done
# done

# echo "Translate-Test Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in translation-et translation-id translation-it translation-sw translation-ta translation-th translation-tr translation-ht translation-qu
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xcopa -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" --model "gpt-35-turbo-deployment" -e melange --chat-prompt -d xcopa
#     done
# done

# echo "Zero-shot Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in et id it sw ta th tr ht qu
#     do

#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xcopa -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" --model "gpt-35-turbo-deployment" -e melange --chat-prompt -d xcopa

#     done
# done

# echo "Starting GPT-4  Evaluation"
# # echo "Monolingual Evaluation"
# # for prompt_name in "plausible_alternatives_discrete"
# # do
# #     for lang in en et id it sw ta th tr ht qu
# #     do
# #         k=8
# #         echo "Running for language $lang and prompt ${prompt_name} and k $k"
# #         python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-4-32k" -e gpt4v2 --chat-prompt -d xcopa
# #     done
# # done

# echo "Translate-Test Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in translation-et translation-id translation-it translation-sw translation-ta translation-th translation-tr translation-ht translation-qu
#     do
#         echo "Running for language $lang and prompt ${prompt_name}"
#         python -m mega.eval_xcopa -p en -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" --model "gpt-4-32k" -e gpt4v2 --chat-prompt -d xcopa -k 8
#     done
# done

# echo "Checking effect of few-shot size on performance"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for lang in en
#     do
#         for k in 0 1 2 4 8 16
#         do
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-35-tunro" -e gpt4v3 --chat-prompt -d xcopa --timeout 30
#         done
#     done
# done

echo "Starting Turbo  Evaluation"
echo "Monolingual Evaluation"
for prompt_name in "plausible_alternatives_discrete"
do
    for lang in ta
    do
        k=8
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gpt-35-turbo" -e gpt4v2 --chat-prompt -d xcopa --timeout 30
    done
done