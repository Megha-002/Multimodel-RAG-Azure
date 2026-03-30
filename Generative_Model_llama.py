import ollama

response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "BTS"}
    ]
)

print("\nAI Response:\n")
print(response['message']['content'])