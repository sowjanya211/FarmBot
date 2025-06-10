import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

load_dotenv('hface.env')
auth_token=os.getenv("HF_TOKEN")

# Load paths
faiss_index_path = os.path.join("knowledge_base", "farming_knowledge.faiss")
text_data_path = os.path.join("knowledge_base", "farming_knowledge.pkl")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load text passages
with open(text_data_path, "rb") as f:
    passages = pickle.load(f)

# Load language model 
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name,auth_token)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "models/Mistral-7B-Instruct-4bit",auth_token,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True,   
)


print("Setting up pipeline...")
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype="auto",
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9
)

def answer_query(query, top_k=3):
    # Embed the query
    query_embedding = embedder.encode([query])

    # Search top_k closest passages
    distances, indices = index.search(query_embedding, top_k)

    # Combine the retrieved passages
    context = ""
    for idx in indices[0]:
        context += passages[idx] + "\n\n"

    # Create the prompt
    prompt = f"""You are an expert farming assistant. 
Use the following context to answer the user's question accurately and clearly.

Context:
{context}

Question: {query}

Answer:"""

    # Generate the answer
    output = chatbot(prompt)[0]['generated_text']

    # return only the generated answer
    return output[len(prompt):].strip()

