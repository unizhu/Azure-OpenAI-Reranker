import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from dotenv import load_dotenv
import hashlib
import pickle
import warnings

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

# Filter warnings
warnings.filterwarnings("ignore", message="A parameter name that contains")

# Ensure the embeddings_cache directory exists
if not os.path.exists('embeddings_cache'):
    os.makedirs('embeddings_cache')

def save_embeddings(file_hash, embeddings):
    with open(os.path.join('embeddings_cache', f"embeddings_{file_hash}.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_hash):
    with open(os.path.join('embeddings_cache', f"embeddings_{file_hash}.pkl"), "rb") as f:
        return pickle.load(f)

def get_file_hash(file):
    hasher = hashlib.md5()
    buf = file.read()
    hasher.update(buf)
    return hasher.hexdigest()

def get_saved_files():
    return [f for f in os.listdir('embeddings_cache') if f.startswith('embeddings_') and f.endswith('.pkl')]

def get_file_name_from_hash(file_hash):
    for f in os.listdir('embeddings_cache'):
        if f.startswith(f'embeddings_{file_hash}'):
            return f.split('_')[1] + '.pdf'
    return None

def load_all_embeddings():
    embeddings = []
    for file in get_saved_files():
        file_hash = file.split('_')[1].split('.')[0]
        embeddings.extend(load_embeddings(file_hash))
    return embeddings

def main():
    st.title("PDF Embedding and Re-ranking with Azure OpenAI and ColBERT")

    # Display the knowledge base status
    st.markdown("**<span style='color:blue'>Knowledge Base: My Knowledge Base</span>**", unsafe_allow_html=True)
    embedded_files = get_saved_files()
    if embedded_files:
        st.write("Embedded Files:")
        for file in embedded_files:
            st.write(get_file_name_from_hash(file.split('_')[1].split('.')[0]))
    else:
        st.write("No embedded files found.")

    embeddings = None  # Initialize embeddings variable

    # Upload PDF
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_pdf:
        file_hash = get_file_hash(uploaded_pdf)
        uploaded_pdf.seek(0)  # Reset file pointer to the beginning

        # Check if embeddings already exist for this file
        try:
            with st.spinner('Loading cached embeddings...'):
                embeddings = load_embeddings(file_hash)
                st.write("Loaded cached embeddings.")
        except FileNotFoundError:
            with st.spinner('Embedding the uploaded file...'):
                with open(os.path.join('embeddings_cache', f"uploaded_file_{file_hash}.pdf"), "wb") as f:
                    f.write(uploaded_pdf.getbuffer())

                # Load and chunk the PDF
                loader = PyPDFLoader(os.path.join('embeddings_cache', f"uploaded_file_{file_hash}.pdf"))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
                docs = text_splitter.split_documents(documents)
                
                # Load the document into Chroma
                embedding_function = AzureOpenAIEmbeddings(model="text-embedding-3-large")
                db = Chroma.from_documents(docs, embedding_function)

                # Extract embeddings and save them to a local file
                embeddings = [doc.page_content for doc in docs]
                save_embeddings(file_hash, embeddings)
                st.write("Saved embeddings to cache.")

    # Select re-ranking method
    method = st.selectbox("Select re-ranking method:", ("Azure OpenAI", "ColBERT"))

    # Enter query and add submit button
    query = st.text_input("Enter your question:")
    if st.button("Submit"):
        with st.spinner('Submitting your query...'):
            if embeddings is None:
                embeddings = load_all_embeddings()
                if not embeddings:
                    st.error("No embeddings found. Please upload a PDF file.")
                    return

            # Create Chroma database from embeddings
            embedding_function = AzureOpenAIEmbeddings(model="text-embedding-3-large")
            db = Chroma.from_texts(embeddings, embedding_function)
            
            # Query the vector store
            docs = db.similarity_search(query)
            
            # Get the document names
            doc_names = ["My Knowledge Base"] * len(docs)
            
            if method == "Azure OpenAI":
                # Re-rank the results using Azure OpenAI API
                response = azure_openai_rerank(query, docs)
                st.write("Re-ranked Results using Azure OpenAI API:")
                for item, doc_name in zip(response, doc_names):
                    st.write(f"File: {doc_name}")
                    st.write(f"Content: {item['content']}")
                    st.write(f"Score: {item['score']}")
                    st.write("---")
            else:
                # Re-rank the results using ColBERT
                colbert_response = colbert_rerank(query, docs)
                st.write("Re-ranked Results using ColBERT:")
                for item, doc_name in zip(colbert_response, doc_names):
                    st.write(f"File: {doc_name}")
                    st.write(f"Content: {item['document']}")
                    st.write(f"Score: {item['score']}")
                    st.write("---")

def azure_openai_rerank(query, docs):
    llm = AzureChatOpenAI(
        azure_deployment="gpt4o",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = [
        (
            "system",
            "You are a Chinese expert relevance ranker. Given a list of documents and a query, your job is to determine how relevant each document is for answering the query. Your output should be a valid JSON array, where each item is an object with 'content' and 'score' fields. The 'content' field should contain the document text, and the 'score' field should be a number from 50.0 to 100.0, with higher scores indicating higher relevance. But don't give me the duplicated result."
        ),
        ("user", f"Query: {query}\n\nDocs: {[doc.page_content for doc in docs]}")
    ]
    
    ai_msg = llm.invoke(messages)
    
    # Process the AI response
    try:
        # Remove any leading/trailing whitespace and non-JSON characters
        cleaned_content = ai_msg.content.strip().lstrip("```json").rstrip("```").strip()
        response_data = json.loads(cleaned_content)
        
        if isinstance(response_data, list) and all('content' in item and 'score' in item for item in response_data):
            return response_data
        else:
            st.error("Unexpected response format. Expected a list of objects with 'content' and 'score' fields.")
            st.error(cleaned_content)
            return []
    except json.JSONDecodeError as e:
        st.error(f"Error parsing AI response: {e}")
        st.error(cleaned_content)
        return []

def colbert_rerank(query, docs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    scores = []

    def maxsim(query_embedding, document_embedding):
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)
        sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)
        avg_max_sim = torch.mean(max_sim_scores, dim=1)
        return avg_max_sim

    query_encoding = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)

    for document in docs:
        document_encoding = tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = model(**document_encoding).last_hidden_state
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document.page_content,
        })

    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)
    return sorted_data

if __name__ == "__main__":
    main()
