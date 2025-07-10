# 🧠 TaughtAI Support Assistant — Demo Chatbot

This project is a **demo chatbot** created to showcase how TaughtAI's **custom GPTs** can be integrated into a website. The assistant is designed for educators and school staff, delivering trauma-informed, inclusive guidance for supporting students with **SEMH**, **SEND**, **ASD**, **ADHD**, and **trauma backgrounds**.

📌 **Note:** This is a single-bot demo for evaluation purposes. The client (TaughtAI) has 6 separate custom GPTs currently hosted within ChatGPT. This demo simulates how one of them would function independently when integrated into a custom frontend/backend environment.

---

## 🔧 Key Features

- ✅ PDF ingestion and vectorization with `pdfplumber`, `OpenAI Embeddings`, and `Pinecone`
- ✅ FastAPI backend for frontend integration
- ✅ Context-aware support responses tailored to SEND/SEMH practices
- ✅ Removes personally identifiable information from queries
- ✅ Follows specific school policies: **Behaviour**, **Safeguarding**, **SEND**, **Communication**

---

## 📦 Requirements

- Python 3.8+
- `openai`
- `langchain`
- `pinecone-client`
- `pdfplumber`
- `fastapi`
- `uvicorn`
- `python-dotenv`

---

## 📚 Future Integration

- ✅ This demo shows how **one GPT** functions outside ChatGPT.
- 🚀 For full integration, each of the **6 custom GPTs** will have its own endpoint or configuration within the TaughtAI platform.
- 🌐 Frontend team can use this as the backend blueprint for embedding the assistant into the web interface.
