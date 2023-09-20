
# for dataset in xcopa xnli indic_xnli pawsx "xstory_cloze" xquad tydiqa indicqa mlqa pawsx xlsum udpos panx wikiann
# # for dataset in "xstory_cloze"
# do
#     for model in "gpt-35-tunro" "gpt-4-32k"
#     do
#         echo $dataset $model
#         python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_method "fill_dataset_card" --save_dir analysis/results/contamination/
#     done
# done


for dataset in xcopa xnli indic_xnli pawsx "xstory_cloze" xquad tydiqa indicqa mlqa pawsx xlsum udpos panx wikiann
# for dataset in "xstory_cloze"
do
    for model in "gpt-35-turbo" "gpt-4-32k"
    do
        echo $dataset $model
        python -m mega.analysis.contamination -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_method "fill_dataset_card_w_example" --save_dir analysis/results/contamination/
    done
done