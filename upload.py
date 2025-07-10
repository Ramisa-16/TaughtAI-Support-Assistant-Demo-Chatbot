import os
import pickle
import pdfplumber
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ------------------ Load API Keys ------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

PINECONE_INDEX_NAME = "bentaughtai-index"

# ------------------ Initialize Pinecone ------------------

pc = Pinecone(api_key=pinecone_api_key)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ------------------ PDF Setup ------------------

pdf_paths = [
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\ASD school stress and anxiety - Autism Educaiotn Trust.pdf",
    r"bentaughtai-attachments/Autism Speaks - Strategies.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\Classroom Strategies Working with EAL Learners in Secondary Settings - Bell Foundation.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\EEF - Improving_Behaviour_in_Schools_Evidence_Review.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\EEF - Metacognition_and_self-regulated_learning.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\EEF - Parental_Engagement_-_Evidence_from_Research_and_Practice.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\EEF - Social_and_Emotional_Learning_Evidence_Review.pdf",
    r"bentaughtai-attachments/EEF - special_educational_needs_in_mainstream_schools_guidance_report_2025-04-10-110432_klxp.pdf",
    r"bentaughtai-attachments/EEF - Teacher_Feedback_to_Improve_Pupil_Learning.pdf",
    r"bentaughtai-attachments/EEF_KS3_KS4_LITERACY_GUIDANCE.pdf",
    r"bentaughtai-attachments/Evidence-based strategies for ADHD.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\Rosenshine - Principles-of-Insruction.pdf",
    r"C:\Users\Ramisa\Desktop\bentaughtai\bentaughtai-attachments\Shire - Teaching-and-Managing-Students-with-ADHD.pdf",
    r"bentaughtai-attachments/TRAUMA-INFORMED INSTRUCTION - Hanover institute.pdf"
]

PROCESSED_DATA_FILE = "processed_data.pkl"

# ------------------ Utilities ------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def process_and_save_pdfs():
    processed_data = {}
    for pdf_file in pdf_paths:
        name = os.path.basename(pdf_file)
        text = extract_text_from_pdf(pdf_file)
        chunks = split_text_into_chunks(text)
        combined_text = " ".join(chunks)
        processed_data[name] = {"text": combined_text}
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(processed_data, f)

def load_and_upsert_data():
    if not os.path.exists(PROCESSED_DATA_FILE):
        process_and_save_pdfs()

    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)

    stats = index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        print("Uploading vectors to Pinecone...")
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        for pdf_name, data in processed_data.items():
            chunks = split_text_into_chunks(data["text"])
            documents = [Document(page_content=text) for text in chunks]
            vectors = []
            for i, doc in enumerate(documents):
                embedding = embeddings.embed_query(doc.page_content)
                vectors.append((
                    f"{pdf_name}_chunk_{i}",
                    embedding,
                    {"text": doc.page_content}
                ))
            for i in range(0, len(vectors), 50):
                index.upsert(vectors=vectors[i:i+50])
        print("Upload complete.")

if __name__ == "__main__":
    load_and_upsert_data()
