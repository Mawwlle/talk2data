import openai
client = openai.OpenAI(
    api_key="sk-vVlZM0J1ViEnD5SFtFWW1g",
    base_url="http://10.32.15.88:4000/v1"
)

response = client.chat.completions.create(
    model="qwen-coder-32b", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
