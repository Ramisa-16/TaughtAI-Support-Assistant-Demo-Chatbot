import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# ------------------ Load API Keys ------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

PINECONE_INDEX_NAME = "bentaughtai-index"
MAX_HISTORY_LENGTH = 10

# ------------------ Initialize Clients ------------------

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(PINECONE_INDEX_NAME)
client = OpenAI(api_key=openai_api_key)

# ------------------ Utilities ------------------

def remove_formatting(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'#{2,}\s*', '', text)
    text = re.sub(r"[\"']the learner[\"']", "the learner", text, flags=re.IGNORECASE)
    return text.strip()

def get_support_response(user_input, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    query_embedding = embeddings.embed_query(user_input)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = " ".join([match.metadata.get('text', '') for match in results.matches])

    system_message = (
        "You are a trauma-informed educational support assistant trained in SEMH, SEND, and inclusive practices. "
        "You support anonymized staff queries with relational, classroom-based strategies tailored to learners with ASD, "
        "ADHD, trauma backgrounds, and behavioural concerns. Responses must follow Taught AI Academy’s Behaviour, "
        "Safeguarding, SEND, and Communication Policies. Never allow personal names or identifiers. "
        "If a name is present in the user’s query, replace it with 'the learner'."
    )

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history[-MAX_HISTORY_LENGTH:],
        {"role": "user", "content": f"User Query: {user_input}\n\nContext:\n{context}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    ai_reply = response.choices[0].message.content.strip()
    ai_reply = remove_formatting(ai_reply)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": ai_reply})

    return ai_reply, conversation_history

# ------------------ Test ------------------

if __name__ == "__main__":
    query = "How can I support Bob with ADHD who gets anxious in noisy classrooms?"
    response, _ = get_support_response(query)
    print("\nResponse:\n", response)