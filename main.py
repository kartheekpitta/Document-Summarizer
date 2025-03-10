
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ✅ Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ✅ Function to load PDF documents from 'data' folder
def load_documents():
    documents = []
    for file in os.listdir('data'):
        if file.endswith('.pdf'):
            loader = PyMuPDFLoader(os.path.join('data', file))
            documents.extend(loader.load())
    return documents

# ✅ Function to split text into manageable chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    return texts

# ✅ Function to create embeddings and FAISS index
def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# ✅ Function to summarize documents using Huggingface
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_documents():
    documents = load_documents()
    if not documents:
        print("⚠️ No PDF documents found in the 'data' folder.")
        return

    texts = split_text(documents)
    vectorstore = create_embeddings(texts)

    # ✅ Batch process summaries to avoid index error
    summary_text = ""
    for text in texts:
        chunk_summary = summarizer(text.page_content, max_length=500, min_length=100, do_sample=False)
        summary_text += chunk_summary[0]['summary_text'] + "\n\n"

    # ✅ Save the summary to a text file
    with open('outputs/summary.txt', 'w') as f:
        f.write(summary_text)

    print("✅ Summary successfully saved to 'outputs/summary.txt'")

if __name__ == "__main__":
    summarize_documents()
