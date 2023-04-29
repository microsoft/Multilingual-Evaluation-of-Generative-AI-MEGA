# for prompt in "C1 or C2? premise, so/because…" "best_option" "best_option_discrete" "cause_effect" "choose" "exercise" "i_am_hesitating" "more likely" "plausible_alternatives"
# for prompt in "C1 or C2? premise, so/because…" "best_option" "cause_effect" "choose" "exercise" "i_am_hesitating" "more likely" "plausible_alternatives"
for prompt in "C1 or C2? premise, so/because…" "best_option" "best_option_discrete" "cause_effect" "choose" "exercise" "i_am_hesitating" "more likely" "plausible_alternatives" "plausible_alternatives_discrete"
do
    echo "Running for Prompt $prompt"
    python -m mega.eval_xcopa -p en -t en --pivot_prompt_name "$prompt" --tgt_prompt_name "$prompt" --eval_on_val -k 8 --model "gpt-35-turbo-deployment" -e melange --chat-prompt -d xcopa
done