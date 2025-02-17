import os
from openai import OpenAI


self_api_key = ''


client = OpenAI(
    base_url="https://api.gpts.vin/v1",
    api_key=self_api_key
)  # 





def generate(prompt):
    try:
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        max_tokens = 100,
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
        
        return completion.choices[0].message.content, completion.usage.total_tokens/1000
    except:
        return 'generate error', 0
    
    
    

def generate_embedding(text):
    model="text-embedding-ada-002"
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    return client.embeddings.create(
            input=[text], model=model).data[0].embedding
    
