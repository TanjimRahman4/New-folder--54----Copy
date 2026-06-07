# main.py
from huggingface_hub import InferenceClient
from config import HF_TOKEN

# Initialize the Hugging Face client with your teacher's token
client = InferenceClient(token=HF_TOKEN)

# Using Llama 3 8B Instruct - a beast at answering anything
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize chat history with a strong system guide
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
        # Get input from you
        user_input = input("You: ")
        
        # Exit condition
        if user_input.strip().lower() in ["quit", "exit"]:
            print("\n🤖 AI: Catch you later!")
            break
            
        # Skip if you just hit enter
        if not user_input.strip():
            continue

        # Save your message to history
        messages.append({"role": "user", "content": user_input})
        print("\nThinking...")

        # Send everything to Hugging Face
        response = client.chat_completion(
            messages=messages,
            model=MODEL_ID,
            max_tokens=1200,
            temperature=0.7,
        )

        # Grab the text reply
        assistant_reply = response.choices[0].message.content
        
        print("-" * 50)
        print(f"AI: {assistant_reply}")
        print("-" * 50 + "\n")

        # Save the AI's reply to history so it remembers the context
        messages.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        print(f"\n❌ error: {e}\n")