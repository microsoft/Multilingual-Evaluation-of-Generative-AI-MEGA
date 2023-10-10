echo "ZS Cross Lingual Evaluation for 3.5 turbo"
for prompt_name in "Answer Given options"
do
    for lang in es eu hi id my ru sw te zh
    # for lang in te
    do
        k=4
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p en -t $lang --model "gpt-35-turbo" --tgt_prompt_name "${prompt_name}" --temperature 0 -k $k --timeout 30 --chat-prompt
    done
done

# echo "ZS Cross Lingual Evaluation for gpt4"
# for prompt_name in "Answer Given options"
# do
#     for lang in ar es eu hi id my ru sw te zh
#     # for lang in te
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v3 -p en -t $lang --model "gpt4" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb -k $k --timeout 30
#     done
# done
