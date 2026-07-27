# Import FAISS library for efficient similarity search on vector embeddings
import faiss
# Import NumPy for numerical operations and array manipulation
import numpy as np
# Import OpenAI client to interact with OpenAI APIs (embeddings and chat models)
from openai import OpenAI
# Import os module to access environment variables and file paths
import os
# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv
# Load environment variables from the .env file into the system environment
load_dotenv()
# Create an OpenAI client using the API key stored in environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path where the FAISS index and chunked documents are stored
VECTOR_PATH = "vectorstore"

# Load the FAISS vector index from the specified directory
index = faiss.read_index("vectorstore/faiss.index")
# Load the stored document chunks (text segments) from a NumPy file
# allow_pickle=True allows loading Python objects stored in the array
chunks = np.load("vectorstore/chunks.npy", allow_pickle=True)


# Function to convert a user query into an embedding vector
def embed_query(query):
    # Call OpenAI embeddings API to generate vector representation for the query
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Embedding model used
        input=query                      # Query text to be converted into embedding
    )
    # Convert the embedding list into a NumPy array of type float32 (required by FAISS)
    return np.array(response.data[0].embedding).astype("float32")

# Function to retrieve top-k relevant document chunks based on query similarity
def retrieve(query, k=2):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results


# Function to send the retrieved context and user question to the LLM
def ask_llm(question, context):
    prompt = f"""
                Use the following context to answer the question. 
                Use only the given context to answer the question. 
                If answer is not present in the context then say "I don't know".

                Context:
                {context}

                Question:
                {question}

                Answer:
            """
    # Call OpenAI Chat Completion API with the constructed prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role":"user","content":prompt}]  
    )
    print("LLM Response:", response)
    return response.choices[0].message.content

# Main function that executes the full RAG pipeline
def ask_question(question):
    # Retrieve relevant document chunks for the question
    docs = retrieve(question)
    # Combine all retrieved chunks into a single context string separated by new lines
    context = "\n".join(docs)
    # Send the question and context to the LLM to generate an answer
    answer = ask_llm(question, context)
    # Return the generated answer
    return answer