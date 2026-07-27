import os
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VECTOR_PATH="vectorstore"

def read_pdf(file_path):
    reader=PdfReader(file_path)
    text=""
    for page in reader.pages:
        text+=page.extract_text()
        
    return text

def chunk_text(text,chunk_size=1000,overlap=200):
    chunks=[]
    step=chunk_size-overlap
    for i in range(0,len(text),step):
        #print(i)
        chunk=text[i:i+chunk_size]
        chunks.append(chunk)
        
    return chunks    

def create_embeddings(chunks):
    embeddings=[]
    for chunk in chunks:
        response=client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        #print("Response:",response)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype("float32")    

def build_index(pdf_path):
    text=read_pdf(pdf_path)
    #print("text:",text)
    chunks=chunk_text(text)
    #print("chunks:",chunks)
    embeddings=create_embeddings(chunks)
    #print("embeddings:",embeddings)
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs("vectorstore",exist_ok=True)
    faiss.write_index(index,"vectorstore/faiss.index")
    np.save("vectorstore/chunks.npy",np.array(chunks))
    print("vector DB created")
    


if __name__=="__main__":
    build_index(r"D:\QA_RAG\VK.pdf")