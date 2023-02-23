# for prompt in "C1 or C2? premise, so/because…" "best_option" "cause_effect" "choose" "exercise" "i_am_hesitating" "more likely" "plausible_alternatives" "…As a result, C1 or C2?" "…What could happen next, C1 or C2?" "…which may be caused by" "…why? C1 or C2"
for prompt in "more likely" "…As a result, C1 or C2?" "…What could happen next, C1 or C2?" "…which may be caused by" "…why? C1 or C2"
do
    echo "Running for Prompt $prompt"
    python -m mega.eval_xcopa -p en -t en --pivot_prompt_name "$prompt" --tgt_prompt_name "$prompt" --eval_on_val --model BLOOM
done