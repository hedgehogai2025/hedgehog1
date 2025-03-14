from openai import OpenAI

# Initialize the client with your API key
client = OpenAI(
    api_key="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"
)

try:
    # Make a minimal API call with the new client format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("API key is valid!")
    print(response)
except Exception as e:
    print("API key verification failed:")
    print(e)