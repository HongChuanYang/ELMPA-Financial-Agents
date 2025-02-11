import os
# os.environ['OPENAI_API_KEY'] = 'AIzaSyBd0viqzhGowR-Jsly6So3_h64audPdy9k'
# import openai
from openai import OpenAI
# import httpx

self_api_key = ''



client = OpenAI(api_key = self_api_key, base_url="https://api.deepseek.com")

def generate(prompt):
    try:
        completion = client.chat.completions.create(
        model="deepseek-chat",
        # max_tokens = 100,
        temperature = 0.5,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0,
        stream = False,
        messages=[
            {"role": "system", "content": "I want you to act as an agent. Please return your simulation results in a JSON format as a single line without any whitespace."},
            {"role": "user", "content": prompt}
        ]
        )
        # print(completion)
        # print(completion.usage)
        
        return completion.choices[0].message.content, completion.usage.total_tokens/1000
    except:
        return 'generate error', 0
    
    

    
