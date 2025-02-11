import os
# os.environ['OPENAI_API_KEY'] = 'AIzaSyBd0viqzhGowR-Jsly6So3_h64audPdy9k'
# import google.generativeai as genai
# import openai
from openai import OpenAI
import httpx


# openai.api_base = 'https://api.gpts.vin/v1'
# openai.api_key = 'sk-dq6VlZSgraqHR8pRB7008dEc9d30426e8aC834F1EaD4DaAf'


client = OpenAI(
    base_url="https://api.xiaoai.plus/v1", 
    api_key="sk-M8ZTAxBXC5mxdW51J9Wlg9oSJPHMv3cfyNhN2aJout0QnXYB",
    http_client=httpx.Client(
        base_url="https://api.xiaoai.plus/v1",
        follow_redirects=True,
    ),
)



def generate(prompt):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens = 100,
    temperature = 0.1,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
    stream = False,
    messages=[
        {"role": "system", "content": "I want you to act as an agent. Please return your simulation results in a JSON format as a single line without any whitespace."},
        {"role": "user", "content": prompt}
    ]
    )
    print(completion.usage)
    return completion.choices[0].message.content





def generate_embedding(text):

    # client = OpenAI(
    #     base_url="https://api.xiaoai.plus/v1", 
    #     api_key="sk-3BhhSOyNFpkUYSNg23D854AeDe0142A39f1337A89258974e",
    #     http_client=httpx.Client(
    #         base_url="https://api.xiaoai.plus/v1",
    #         follow_redirects=True,
    #     ),
    # )
    # openai embedding are charged. only use with 贵的直连
    model="text-embedding-ada-002"
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    return client.embeddings.create(
            input=[text], model=model).data[0].embedding


