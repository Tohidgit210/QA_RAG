QA_RAG/
│
├── backend/
│   ├── main.py
│   ├── ingest.py
│   └── rag.py
│
├── vectorstore/
│
├── frontend/
│   └── app.py
│
├── requirements.txt
└── .env


python -m venv venv

source /Users/shreekantjere/MIT_QA_Systems/QA_RAG/venv/bin/activate

.\venv\Scripts\activate

uvicorn backend.main:app --reload

streamlit run frontend/app.py



# Swagger URL
http://localhost:8000/docs

# Curl Command
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "when was Virat Kohli born?"
}'



# What is IndexFlatL2?
# IndexFlatL2 is a simple FAISS index structure that:
# stores vectors exactly as they are
# performs brute-force similarity search
# calculates L2 distance (Euclidean distance) between vectors
# It is called Flat because no compression or clustering is used.




# How reshape(1, -1) Works?
reshape(rows, columns)
Value	Meaning
 1	    Force the array to have 1 row
-1	    NumPy automatically calculates the number of columns