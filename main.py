from huggingface_hub import InferenceClient
from config import HF_TOKEN

client = InferenceClient(token=HF_TOKEN)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

messages = [
    {
        "role": "system", 
        "content": "You are a highly intelligent, precise, and helpful AI assistant. You can answer any questions flawlessly."
    }
]

print("=" * 50)
print("🤖 Hugging Face AI Chatbot Initialized!")
print("Type 'quit' or 'exit' to close the chat.")
print("=" * 50 + "\n")

while True:
    try:
        user_input = input("You: ")
        
        if user_input.strip().lower() in ["quit", "exit"]:
            print("\n🤖 AI: Catch you later!")
            break
            
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        print("\nThinking...")

        response = client.chat_completion(
            messages=messages,
            model=MODEL_ID,
            max_tokens=1200,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content
        
        print("-" * 50)
        print(f"AI: {assistant_reply}")
        print("-" * 50 + "\n")

        messages.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        print(f"\n❌ error: {e}\n")