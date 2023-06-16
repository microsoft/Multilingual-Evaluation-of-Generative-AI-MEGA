# echo "Starting Turbo  Evaluation"
# echo "Monolingual Evaluation"
# for prompt_name in "Answer Given options"
# do
#     for lang in en ar es eu hi id my ru sw te zh
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model gpt-35-tunro --tgt_prompt_name "${prompt_name}" --temperature 0 --chat-prompt --log_wandb
#     done
# done

# echo "Translate-Test Evaluation"
# for prompt_name in "Answer Given options"
# do
#     for lang in ar es eu hi id my ru sw te zh
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p en -t $lang --model gpt-35-tunro --tgt_prompt_name "${prompt_name}" --temperature 0 --chat-prompt --log_wandb --translate-test
#     done
# done


# echo "Starting DV003  Evaluation"

# echo "Prompt Tuning"

# for prompt_name in "Answer Given options" "Choose Story Ending" "Movie What Happens Next" "Story Continuation and Options" "Novel Correct Ending"
# do
#     for lang in en
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model "text-davinci-003" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb --eval_on_val
#     done
# done


# echo "Monolingual Evaluation"
# for prompt_name in "Answer Given options"
# do
#     # for lang in en ar es eu hi id my ru sw te zh
#     for lang in my
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model "text-davinci-003" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb
#     done
# done

# echo "ZS Cross Lingual Evaluation for turbo"
# for prompt_name in "Answer Given options"
# do
#     for lang in en ar es eu hi id my ru sw te zh
#     # for lang in te
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model "gpt-35-tunro" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb -k $k
#     done
# done

# echo "ZS Cross Lingual Evaluation for dv003"
# for prompt_name in "Answer Given options"
# do
#     for lang in en ar es eu hi id my ru sw te zh
#     # for lang in te
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model "text-davinci-003" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb -k $k
#     done
# done

# echo "Translate-Test Evaluation"
# for prompt_name in "Answer Given options"
# do
#     # for lang in ar es eu hi id my ru sw te zh
#     for lang in eu hi id my ru sw te zh
#     do
#         k=4
#         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#         python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p en -t $lang --model "text-davinci-003" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb --translate-test -k $k
#     done
# done

echo "Starting GPT-4 Evaluation"
echo "Monolingual Evaluation"
for prompt_name in "Answer Given options"
do
    # for lang in en ar es eu hi id my ru sw te zh
    for lang in en es eu id ru sw te
    do
        k=4
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model gpt-4-32k --tgt_prompt_name "${prompt_name}" --temperature 0 --chat-prompt --log_wandb
    done
done

echo "Translate-Test Evaluation"
for prompt_name in "Answer Given options"
do
    # for lang in ar es eu hi id my ru sw te zh
    for lang in hi ru zh
    do
        k=4
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p en -t $lang --model gpt-4-32k --tgt_prompt_name "${prompt_name}" --temperature 0 --chat-prompt --log_wandb --translate-test
    done
done