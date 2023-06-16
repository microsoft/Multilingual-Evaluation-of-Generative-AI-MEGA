import os
import openai
from mega.utils.env_utils import load_env
from mega.models.completion_models import gpt3x_completion
from tqdm import tqdm
import json
import time
openai.api_type = "azure"
openai.api_base = "https://gcrgpt4aoai6c.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


load_env(env_name="gpt4v2")

# response = openai.ChatCompletion.create(
#     engine="gpt-4-32k",
#     messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"We are collecting more examples for a COPA dataset which will be used to test a system\\'s ability of Commonsense Causal Judgments.\\n\\nThe format of the data:\\n A premise: which is a statement of something happened, and two choices that could plausibly happen as the RESULT of the premise. The correct choice is the alternative that is MORE plausible than the wrong choice.\\n\\n Here are some examples of the data in Tamil: \\n\\nExample 1:\\n{\"premise\": \"அந்த மாணவி அந்த சொல்லைத் தவறாக எழுத்துக் கூட்டினாள் What happened as a RESULT?\", \"correct choice\": \"ஆசிரியர் அவளின் பிழையைத் திருத்தினார்\", \"wrong choice\": \"ஆசிரியர் அவள்மீது அலட்சியம் காட்டினார்\"}\\n\\nExample 2:\\n{\"premise\": \"நான் பாரவண்டியைத் தள்ளினேன் What happened as a RESULT?\", \"correct choice\": \"பாரவண்டியின் சக்கரங்கள் முன்னே உருண்டோடின\", \"wrong choice\": \"பாரவண்டியிலிருந்த பொருட்கள் வெளியே விழுந்தன\"}\\n\\nExample 3:\\n{\"premise\": \"நான் ஈரப் பைஞ்சுதையில் என் கையைப் பதித்தேன் What happened as a RESULT?\", \"correct choice\": \"எனது கைரேகை பைஞ்சுதையில் உலர்ந்தது\", \"wrong choice\": \"பைஞ்சுதையில் விரிசல் விழுந்தது\"}\\n\\nExample 4:\\n{\"premise\": \"அந்த பெண்மணி பணி ஓய்வு பெற்றாள் What happened as a RESULT?\", \"correct choice\": \"அவள் தனது ஓய்வூதியத்தைப் பெற்றுக் கொண்டாள்\", \"wrong choice\": \"அவள் அடமானம் செய்த சொத்துக்களைப் பணம் கட்டி மீட்டெடுத்தாள்\"}\\n\\nExample 5:\\n{\"premise\": \"நோய்வாய்ப்பட்ட குழந்தை தன் நண்பனின் மீது இருமியது What happened as a RESULT?\", \"correct choice\": \"அதன் நண்பன் நோய்வாய்ப்பட்டான்\", \"wrong choice\": \"அதன் நண்பன் தும்மினான்\"}\\n\\nGenerate 10 new examples in Tamil"},{"role":"assistant","content":"This model's maximum context length is 32768 tokens. However, you requested 35938 tokens (5720 in the messages, 30218 in the completion). Please reduce the length of the messages or completion. | Apim-request-id: 05ca6c6a-4faa-40c5-ad24-06037a26652b"}],
#     temperature=0.5,
#     max_tokens=30218,
#     top_p=0.95,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None  
# )
prompt = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"We are collecting more examples for a COPA dataset which will be used to test a system\\'s ability of Commonsense Causal Judgments.\\n\\nThe format of the data:\\n A premise: which is a statement of something happened, and two choices that could plausibly happen as the RESULT of the premise. The correct choice is the alternative that is MORE plausible than the wrong choice.\\n\\n Here are some examples of the data in Tamil: \\n\\nExample 1:\\n{\"premise\": \"அந்த மாணவி அந்த சொல்லைத் தவறாக எழுத்துக் கூட்டினாள் What happened as a RESULT?\", \"correct choice\": \"ஆசிரியர் அவளின் பிழையைத் திருத்தினார்\", \"wrong choice\": \"ஆசிரியர் அவள்மீது அலட்சியம் காட்டினார்\"}\\n\\nExample 2:\\n{\"premise\": \"நான் பாரவண்டியைத் தள்ளினேன் What happened as a RESULT?\", \"correct choice\": \"பாரவண்டியின் சக்கரங்கள் முன்னே உருண்டோடின\", \"wrong choice\": \"பாரவண்டியிலிருந்த பொருட்கள் வெளியே விழுந்தன\"}\\n\\nExample 3:\\n{\"premise\": \"நான் ஈரப் பைஞ்சுதையில் என் கையைப் பதித்தேன் What happened as a RESULT?\", \"correct choice\": \"எனது கைரேகை பைஞ்சுதையில் உலர்ந்தது\", \"wrong choice\": \"பைஞ்சுதையில் விரிசல் விழுந்தது\"}\\n\\nExample 4:\\n{\"premise\": \"அந்த பெண்மணி பணி ஓய்வு பெற்றாள் What happened as a RESULT?\", \"correct choice\": \"அவள் தனது ஓய்வூதியத்தைப் பெற்றுக் கொண்டாள்\", \"wrong choice\": \"அவள் அடமானம் செய்த சொத்துக்களைப் பணம் கட்டி மீட்டெடுத்தாள்\"}\\n\\nExample 5:\\n{\"premise\": \"நோய்வாய்ப்பட்ட குழந்தை தன் நண்பனின் மீது இருமியது What happened as a RESULT?\", \"correct choice\": \"அதன் நண்பன் நோய்வாய்ப்பட்டான்\", \"wrong choice\": \"அதன் நண்பன் தும்மினான்\"}\\n\\nGenerate 10 new examples in Tamil"}]


for _ in tqdm(range(3000)):
    response = gpt3x_completion(
        prompt = prompt,
        model = "gpt-4-32k",
        temperature = 0.5,
        max_tokens=1000
    )
    
    
    try:
        with open("results/tamil_copa.json", "r") as f:
            responses = json.load(f)
    except:
        responses = []
        
    responses.append(response)

    with open("results/tamil_copa.json", "w") as f:
        json.dump(obj=responses, fp =f, ensure_ascii=False, indent=4)
        
    time.sleep(10)

    

