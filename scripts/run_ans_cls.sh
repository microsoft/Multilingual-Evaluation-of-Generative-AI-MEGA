echo "Monolingual Evaluation"
for lang in bn ko sw id ar en fi te ru
do
    for k in 0 4
    do
        echo "Running for language $lang and k=$k"
        python -m mega.answer_cls -d tydiqa -p $lang -t $lang --eval_on_val -k $k
    done
done

echo "Zero-Shot Evaluation"
for lang in bn ko sw id ar fi te ru
do
    for k in 0 4
    do
        echo "Running for language $lang and k=$k"
        python -m mega.answer_cls -d tydiqa -p en -t $lang --eval_on_val -k $k
    done
done