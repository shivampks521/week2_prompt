# import os
# from dotenv import load_dotenv
# import  google.generativeai as genai
# import pytesseract
# import pdfplumber
# import chromadb
# from chromadb.utils import embedding_functions
# from google.cloud import aiplatform
# from sentence_transformers import SentenceTransformer  # Local embedding model
# from typing import List  # Fix for embedding function

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  #takes api key

# local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def local_embedding_function(input: List[str]) -> List[List[float]]:
#     return local_model.encode(input, convert_to_list=True)



# chroma_client = chromadb.PersistentClient(path= "chroma_persistent_storage")
# collection_name= "document_qa_collection"
# collection= chroma_client.get_or_create_collection(name= collection_name, embedding_function=local_embedding_function)

# chat_model = genai.GenerativeModel("gemini-1.5-flash")

# response = chat_model.generate_content("What is human life expectancy in India?")
# print(response.text)



# import os
# from dotenv import load_dotenv
# import google.generativeai as genai
# import pytesseract
# import pdfplumber
# import chromadb
# from chromadb.utils import embedding_functions
# from google.cloud import aiplatform
# from sentence_transformers import SentenceTransformer  # Local embedding model
# from typing import List

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Takes API key

# local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Define a custom embedding function class (without subclassing DefaultEmbeddingFunction)
# class LocalEmbeddingFunction:
#     def __call__(self, input: List[str]) -> List[List[float]]:
#         return local_model.encode(input, convert_to_list=True)

# # Instantiate the embedding function
# local_embedding_function = LocalEmbeddingFunction()

# chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
# collection_name = "document_qa_collection"

# # Pass the class instance as the embedding function
# collection = chroma_client.get_or_create_collection(
#     name=collection_name,
#     embedding_function=local_embedding_function
# )

# chat_model = genai.GenerativeModel("gemini-2.0-flash")

# response = chat_model.generate_content("What is human life expectancy in India?")
# print(response.text)




# def load_documents_from_directory(directory_path):
#     print("___________loading the directory_______________")
#     documents=[]

#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path,filename)

#         if filename.endswith(".pdf"):
#             with pdfplumber.open(file_path) as pdf:
#                 text= "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#                 documents.append({"id": filename, "text":text})

#     return documents
        

# def split_text(text, chunk_size= 1024, chunk_overlap=40):
#     chunks=[]
#     start= 0
#     while start < len(text):
#         end= start+ chunk_size
#         chunks.append(text[start:end])
#         start = end- chunk_overlap
#     return chunks

# directory_path= "/home/shivam/git_practice/prompt/"

# documents= load_documents_from_directory(directory_path)


# for doc in documents:
#     text_chunks= split_text(doc["text"])
#     for i, chunk in enumerate(text_chunks):
#         collection.add(
#             ids=[f"{doc['id']}_chunks_"],
#             embeddings= [local_model.encode(chunk,convert_to_list= True)],
#             documents=[chunk]
#         )

# print("document uploaded")


# # Function to retrieve relevant chunks from ChromaDB
# def query_documents(query, top_k=5):
#     query_embedding = local_model.encode([query], convert_to_list=True)[0]
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     relevant_chunks = results["documents"][0] if "documents" in results else []
#     return relevant_chunks

# # Function to generate a response using Gemini-1.5
# def generate_response(question, relevant_chunks):
#     context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."

#     prompt = (
#             "You are an AI legal assistant specializing in Indian law, specifically the Bhartiya Nyay Sanhita (BNS). "
#              "Use the retrieved context from the BNS document to provide legally accurate and well-informed responses. "
#     "If the retrieved context is insufficient, offer a general legal perspective based on Indian law, "
#      "but clearly state any uncertainty. Ensure your response is concise, factual, and aligned with legal provisions.\n\n"

#         "Context:\n" + context + "\n\nQuestion:\n" + question
#     )

#     chat_model = genai.GenerativeModel("gemini-1.5-flash")
#     response = chat_model.generate_content(prompt)

#     return response.text

# # Example query and response generation
# question = "\In every case in which an offender is punishable with imprisonment which may be of either description, it shall be competent to the Court which sentences such offender to direct in the sentence that such imprisonment shall be wholly rigorous, or that such imprisonment shall be wholly simple, or that any part of such imprisonment shall be rigorous and the rest simple."
# relevant_chunks = query_documents(question)
# answer = generate_response(question, relevant_chunks)

# print("🔍 Query:", question)
# # print("📄 Retrieved Context:\n", relevant_chunks)
# print("💡 AI Response:\n", answer)



# import os
# from dotenv import load_dotenv
# import google.generativeai as genai
# import pytesseract
# import pdfplumber
# import chromadb
# from chromadb.utils import embedding_functions
# from google.cloud import aiplatform
# from sentence_transformers import SentenceTransformer  # Local embedding model
# from typing import List
# from openai import OpenAI


# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Takes API key

# local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Define a custom embedding function class (without subclassing DefaultEmbeddingFunction)
# class LocalEmbeddingFunction:
#     def __call__(self, input: List[str]) -> List[List[float]]:
#         return local_model.encode(input, convert_to_list=True)

# # Instantiate the embedding function
# local_embedding_function = LocalEmbeddingFunction()

# chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
# collection_name = "document_qa_collection"

# # Pass the class instance as the embedding function
# collection = chroma_client.get_or_create_collection(
#     name=collection_name,
#     embedding_function=local_embedding_function
# )

# chat_model = genai.GenerativeModel("gemini-2.0-flash")

# response = chat_model.generate_content("What is human life expectancy in India?")
# print(response.text)




# def load_documents_from_directory(directory_path):
#     print("___________loading the directory_______________")
#     documents=[]

#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path,filename)

#         if filename.endswith(".pdf"):
#             with pdfplumber.open(file_path) as pdf:
#                 text= "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#                 documents.append({"id": filename, "text":text})

#     return documents
        

# def split_text(text, chunk_size= 1024, chunk_overlap=40):
#     chunks=[]
#     start= 0
#     while start < len(text):
#         end= start+ chunk_size
#         chunks.append(text[start:end])
#         start = end- chunk_overlap
#     return chunks

# directory_path= "/home/shivam/git_practice/prompt/"

# documents= load_documents_from_directory(directory_path)


# for doc in documents:
#     text_chunks= split_text(doc["text"])
#     for i, chunk in enumerate(text_chunks):
#         collection.add(
#             ids=[f"{doc['id']}_chunks_"],
#             embeddings= [local_model.encode(chunk,convert_to_list= True)],
#             documents=[chunk]
#         )

# print("document uploaded")


# # Function to retrieve relevant chunks from ChromaDB
# def query_documents(query, top_k=5):
#     query_embedding = local_model.encode([query], convert_to_list=True)[0]
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     relevant_chunks = results["documents"][0] if "documents" in results else []
#     return relevant_chunks

# # Function to generate a response using Gemini-1.5
# def generate_response(question, relevant_chunks):
#     context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."

#     prompt = (
#             "You are an AI legal assistant specializing in Indian law, specifically the Bhartiya Nyay Sanhita (BNS). "
#              "Use the retrieved context from the BNS document to provide legally accurate and well-informed responses. "
#     "If the retrieved context is insufficient, offer a general legal perspective based on Indian law, "
#      "but clearly state any uncertainty. Ensure your response is concise, factual, and aligned with legal provisions.\n\n"

#         "Context:\n" + context + "\n\nQuestion:\n" + question
#     )

#     chat_model = genai.GenerativeModel("gemini-1.5-flash")
#     response = chat_model.generate_content(prompt)

#     return response.text

# # Example query and response generation
# question = "\In every case in which an offender is punishable with imprisonment which may be of either description, it shall be competent to the Court which sentences such offender to direct in the sentence that such imprisonment shall be wholly rigorous, or that such imprisonment shall be wholly simple, or that any part of such imprisonment shall be rigorous and the rest simple."
# relevant_chunks = query_documents(question)
# answer = generate_response(question, relevant_chunks)

# print("🔍 Query:", question)
# # print("📄 Retrieved Context:\n", relevant_chunks)
# print("💡 AI Response:\n", answer)


import os
from dotenv import load_dotenv
import google.generativeai as genai
import pytesseract
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from google.cloud import aiplatform
from sentence_transformers import SentenceTransformer  # Local embedding model
from typing import List
from openai import OpenAI

# TAKING THE .ENV FILE 
load_dotenv()

# GEMINI KEY THOUGH WE ARE NOT USING IT
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# THIS IS FREE EMBEDDING MODEL( WHICH CONVERTS THE TEXT TO VECTOR INTEGER)
local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# THIS FUNCTION WILL ENCODE THE CONVERTED TEXT INTEGER
class LocalEmbeddingFunction:
    def __call__(self, input: List[str]) -> List[List[float]]:
        return local_model.encode(input, convert_to_list=True)

local_embedding_function = LocalEmbeddingFunction()

# WE ARE USING THE CHROMADB
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=local_embedding_function
)

# WE ARE USING THE DEEPSEEK 
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")  # Store API key securely
)

# Function to load documents from a directorY
def load_documents_from_directory(directory_path):
    print("___________loading the directory_______________")
    documents = []
    #THIS WILL SEARCH FOR THE DIRECTORY PATH 
    for filename in os.listdir(directory_path):
        print("1__________________")
        file_path = os.path.join(directory_path, filename)
        print(f"{file_path}")
        #IF FILE NAME ENDS WITH PDF (WE ARE JUST USING THE ONE PDF) 
        if filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:   #READING THE PDF AS PDF
                print("2_____________________")
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()]) # THIS WILL EXTRACT THE TEXT FROM EACH PAGE  AND WILL ALSO JOIN THE TEXT FROM TEH NEXT PDF AND STORE IT ON TEXT
                print("3_______________________________")
                documents.append({"id": filename, "text": text})

    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1024, chunk_overlap=40):  #HERE SIZE OF CHUNK IS 1024
    chunks = []
    start = 0
    while start < len(text): #START MUST BE LESS THEN TEXT
        print("4__________________________")
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
        print(f"{start}")
    return chunks

# Load and process documents
directory_path = "/home/shivam/git_practice/prompt/"
documents = load_documents_from_directory(directory_path)

for doc in documents:
    text_chunks = split_text(doc["text"]) 
    print("5_______________________________")
    for i, chunk in enumerate(text_chunks):
        collection.add(
            ids=[f"{doc['id']}_chunk_{i}"],
            embeddings=[local_model.encode(chunk, convert_to_list=True)],
            documents=[chunk]
        )

print("📄 Documents uploaded successfully.")

# Function to query documents from ChromaDB
# def query_documents(query, top_k=5):  #ONLY SEARCH FOR THE TOP 5 ANSWERS
#     query_embedding = local_model.encode([query], convert_to_list=True)[0] 
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )
#     print("6_________________________")

#     relevant_chunks = results["documents"][0] if "documents" in results else []
#     return relevant_chunks

# Function to generate a response using DeepSeek AI (via NVIDIA API)
def generate_response(question, relevant_chunks):
    print("7_____________________________")
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."

    prompt = (
        "You are an AI legal assistant specializing in Indian law, specifically the Indian Constitution. "
        "Use the retrieved context from the Indian Constitution to provide legally accurate and well-informed responses. "
        "If the retrieved context is insufficient, offer a general legal perspective based on Indian constitution, "
        "but clearly state any uncertainty. Ensure your response is concise, factual, and aligned with constitutional provisions.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    # Query NVIDIA API (DeepSeek Model)
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )

    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print("8______________________")
            response_text += chunk.choices[0].delta.content

    return response_text

# Example Query and Response
# question = "What is the punishment for wrongful confinement under the Bhartiya Nyay Sanhita (BNS)?"
question="tell me more about Samatha v. State of Andhra Pradesh(1997)"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("\n🔍 Query:", question)
print("\n💡 AI Response:\n", answer)
