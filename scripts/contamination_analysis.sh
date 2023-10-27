#!/bin/bash

# echo Running Contamination Checks for XNLI

# for lang in hi
# do
#     for model in "gpt-35-tunro" "gpt-4-32k"
#     do
#         for contam in "generate" "complete"
#         do
#             echo $lang $model $contam
#             python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d xnli --contam_lang ${lang} --max_tokens 500 --contam_method ${contam} --save_dir analysis/results/contamination/
#         done
#     done
# done

echo Running Contamination Checks for XCOPA

for lang in en et ht it id qu th tr sw
do
    for model in "gpt-35-tunro" "gpt-4-32k"
    do
        for contam in "generate" "complete"
        do
            echo $lang $model $contam
            python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d xcopa --contam_lang ${lang} --max_tokens 1000 --contam_method ${contam} --save_dir analysis/results/contamination/
        done
    done
done

echo Running Contamination Checks for XQUAD

for lang in en ar el hi ru th tr vi de es zh ro
do
    for model in "gpt-35-tunro" "gpt-4-32k"
    do
        for contam in "generate" "complete"
        do
            echo $lang $model $contam
            python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d xquad --contam_lang ${lang} --max_tokens 1000 --contam_method ${contam} --save_dir analysis/results/contamination/
        done
    done
done

echo Running Contamination Checks for TyDiQA
for lang in en ar bn "fi" id ko ru sw te
do
    for model in "gpt-35-tunro" "gpt-4-32k"
    do
        for contam in "generate" "complete"
        do
            echo $lang $model $contam
            python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d tydiqa --contam_lang ${lang} --max_tokens 1000 --contam_method ${contam} --save_dir analysis/results/contamination/
        done
    done
done