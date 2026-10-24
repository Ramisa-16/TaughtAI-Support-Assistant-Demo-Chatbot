<h1 align="center">🎓 TaughtAI Support Assistant</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-00b39e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Live-taughtai.app-success?style=for-the-badge"/>
</p>

<p align="center">
  AI support assistant for the TaughtAI platform — providing intelligent student support,
  answering curriculum questions, and guiding learners through educational content.
  Powers <a href="https://taughtai.app">taughtai.app</a>.
</p>

---

## ✨ Features
- **Multi-School Support** — Taught AI Academy, Primary School, Special School
- **Curriculum-Aware** — answers based on uploaded course materials
- **Student Guidance** — lesson navigation and quiz assistance
- **Teacher Controls** — teachers manage knowledge base via document upload
- **Secure Login** — role-based access (Admin / Teacher / Student)

---

## 🏗️ Architecture

```
Student Query
    │
    ▼
Query Embedding (OpenAI)
    │
    ▼
Pinecone Vector Search
(retrieve relevant curriculum content)
    │
    ▼
GPT-4 / Gemini Response Generation
(grounded in retrieved content)
    │
    ▼
School-specific answer returned
```

---

## 🛠️ Tech Stack
**Python** · **OpenAI GPT** · **Google Gemini** · **Pinecone** · **Django** · **AWS** · **GCP**

---

## 👩‍💻 Author
**Ramisa Sharar Nidhi** — AI Developer (AI Layer) @ JVAI / Betopia Group
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ramisashararnidhi-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/ramisashararnidhi)
