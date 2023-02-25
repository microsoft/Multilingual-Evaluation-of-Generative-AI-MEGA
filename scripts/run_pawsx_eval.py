import os

print( "Monolingual Evaluation")
for prompt_name in ["Concatenation"]: #["Bing-Translated Concatenation"]:
	for lang in ["de"]:#,"es","fr","ja","ko","zh"]:
		for k in [8]:
			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p {lang} -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" --num_proc 24 -k {k}')

# print( "Translate-Test Evaluation")
# for prompt_name in ["Concatenation"]:
# 	for lang in ["de","es","fr","ja","ko","zh"]:
# 		for k in [8]:
# 			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
# 			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" --num_proc 24 -k {k} --translate-test')

# print( "Zero-shot Evaluation")
# for prompt_name in ["Concatenation"]:
# 	for lang in ["de","es","fr","ja","ko","zh"]:
# 		for k in [8]:
# 			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples with Handcrafted Prompts")
# 			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" --num_proc 24 -k {k}')