import os
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Environment variable GROQ_API_KEY is not set.")

client = Groq(api_key=api_key)

result = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ],
    temperature=0.2,
    max_tokens=512,
)

print(result.choices[0].message.content)
