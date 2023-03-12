import os

print( "Monolingual Evaluation")
# for prompt_name in ["Concatenation", "Bing-Translated Concatenation"]:
for prompt_name in ["Bing-Translated Concatenation"]:
    for lang in ["de","es","fr","zh"]:
        for k in [8]:
            print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
            os.system(f'python -m mega.eval_pawsx -d "paws-x" -p {lang} -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model BLOOMZ --num_evals_per_sec 10')

print( "Translate-Test Evaluation")
for prompt_name in ["Concatenation"]:
	for lang in ["de","es","fr","ja","ko","zh"]:
		for k in [8]:
			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --translate-test --model BLOOMZ --num_evals_per_sec 10')

print( "Zero-shot Evaluation")
for prompt_name in ["Concatenation"]:
	for lang in ["de","es","fr","ja","ko","zh"]:
		for k in [8]:
			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples with Handcrafted Prompts")
			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model BLOOMZ --num_evals_per_sec 10')