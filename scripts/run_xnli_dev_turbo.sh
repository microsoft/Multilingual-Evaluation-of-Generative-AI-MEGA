# for prompt_name in "MNLI crowdsource" "based on the previous passage"
for prompt_name in "GPT-3 style" "MNLI crowdsource" "based on the previous passage" "always/sometimes/never"
do
    for temperature in 0
    do
        for lang in en
        do
            echo "Running for language $lang and prompt ${prompt_name} and temperature $temperature"
            python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model "gpt-35-turbo-deployment" --chat-prompt --eval_on_val --test_frac 0.2 --temperature $temperature
        done
    done
done