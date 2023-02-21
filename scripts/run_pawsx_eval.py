import os

print( "Monolingual Evaluation")
for prompt_name in ["handcrafted french concatenation", "handcrafted french meaning", "handcrafted french rewrite"]:
	for k in [8, 4]:
		print(f"Running for {prompt_name} and {k} few-shot examples")
		os.system(f'python -m mega.eval_pawsx -d "paws-x" -p fr -t fr --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" --num_proc 24 -k {k} --test_frac 0.1 --eval_on_val')

print( "Translate-Test Evaluation")
for prompt_name in ["Concatenation", "Meaning" ,"Rewrite"]:
	for k in [8, 4]:
		print(f"Running for {prompt_name} and {k} few-shot examples")
		x = prompt_name.lower()
		os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t fr --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "English {x}" --num_proc 24 -k {k} --translate-test --test_frac 0.1 --eval_on_val')

print( "Zero-shot Evaluation")
for prompt_name in ["Concatenation", "Meaning", "Rewrite"]:
	for k in [8, 4]:
		x = prompt_name.lower()
		print(f"Running for {prompt_name} and {k} few-shot examples with Handcrafted Prompts")
		os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t fr --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "English {x}" --num_proc 24 -k {k} --test_frac 0.1 --eval_on_val')