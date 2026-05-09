from openai import OpenAI

def ask_ai(prompt, api_key):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    completion = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
