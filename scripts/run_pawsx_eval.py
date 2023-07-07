import os

# print("Prompt Tuning")
# for prompt_name in ["Concatenation", "Meaning", "PAWS-ANLI GPT3", "Rewrite", "context-question", "paraphrase-task"]:
# 	for lang in ["en"]:
# 		for k in [8]:
# 			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
# 			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p {lang} -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model "gptturbo" -e vellm --chat-prompt --eval_on_val --test_frac 0.25')

# print( "Monolingual Evaluation")
# # for prompt_name in ["Concatenation", "Bing-Translated Concatenation"]:
# for prompt_name in ["PAWS-ANLI GPT3"]:
#     for lang in ["de","es","fr","ja","ko","zh"]:
#         for k in [8]:
#             print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
#             os.system(f'python -m mega.eval_pawsx -d "paws-x" -p {lang} -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model "gptturbo" -e vellm --chat-prompt')

# print( "Translate-Test Evaluation")
# for prompt_name in ["PAWS-ANLI GPT3"]:
# 	for lang in ["de","es","fr","ja","ko","zh"]:
# 		for k in [8]:
# 			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
# 			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --translate-test --model "gptturbo" -e vellm --chat-prompt')

# print( "Zero-shot Evaluation")
# for prompt_name in ["PAWS-ANLI GPT3"]:
# 	for lang in ["de","es","fr","ja","ko","zh"]:
# 		for k in [8]:
# 			print(f"Running for {lang}, {prompt_name} and {k} few-shot examples with Handcrafted Prompts")
# 			os.system(f'python -m mega.eval_pawsx -d "paws-x" -p en -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model "gptturbo" -e vellm --chat-prompt')

print("Evaluating GPT-4")
print("Monolingual Evaluation")
# for prompt_name in ["Concatenation", "Bing-Translated Concatenation"]:
for prompt_name in ["PAWS-ANLI GPT3"]:
    for lang in ["en", "de", "es", "fr", "ja", "ko", "zh"]:
        for k in [8]:
            print(f"Running for {lang}, {prompt_name} and {k} few-shot examples")
            os.system(
                f'python -m mega.eval_pawsx -d "paws-x" -p {lang} -t {lang} --pivot_prompt_name "{prompt_name}" --tgt_prompt_name "{prompt_name}" -k {k} --model "gpt-4-32k" -e gpt4v2 --chat-prompt'
            )
