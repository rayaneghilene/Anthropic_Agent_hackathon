from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import json
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from glob import glob
from langchain.prompts import ChatPromptTemplate



# system_message = """You are an AI assistant designed to collect specific information from the user in a structured and conversational manner. Your goal is to extract the following details:
# 	•	Age: Ask for the user's age or age range. If they are uncomfortable sharing an exact number, accept a general range.
# 	•	Names: Request their name or preferred name. Allow for multiple names if necessary.
# 	•	Specialization: Ask about their field of expertise, academic background, or professional focus.
# 	•	Topics of Interest: Inquire about subjects they are currently interested in, studying, or working on.
# 	•	Hobbies: Gather information about their leisure activities and interests outside of work/study.
# 	•	Preferred Way of Learning: Ask how they best absorb new information—through reading, videos, hands-on practice, structured courses, etc. 
# Maintain a friendly and engaging tone, adapt to the user's responses dynamically, and encourage them to provide details without making the conversation feel like a survey. If the user provides incomplete answers, gently prompt them for more details."""
system_message = "You are an AI assistant that extracts structured information from text, based on the Json Schema provided generate a prompt for an LLM to perform an action."


schema = """{ 
    "Age": [],
    "Names": [],
    "specialization": [],
    "Topics": [],
    "Hobbies": [],
    "preffered way of learning": []
}"""

app = FastAPI(title="AI Assistant API")

# Load LLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
model.eval()

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('ml' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def predict_NuExtract(model, tokenizer, text, schema, example=["","",""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"

    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to(device)

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]


# Folder to store PDFs
PDF_FOLDER = "Anthropic_Agent_hackathon/PDF_Documents"

# Load PDFs into vectorstore
def load_pdf():
    pdf_files = glob(f"{PDF_FOLDER}/*.pdf")
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-V2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    return index

index = load_pdf()

# Define Retrieval Chain
chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="deepseek-r1:7b"),
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question",
)

# Define schema model
class ExtractionRequest(BaseModel):
    text: str
    schema: str = """{
        "Age": [],
        "Names": [],
        "specialization": [],
        "Topics": [],
        "Hobbies": [],
        "preferred way of learning": []
    }"""
    example: List[str] = ["", "", ""]

# Define chat request model
class ChatRequest(BaseModel):
    question: str


import re

def remove_think_tags(text):
    """Remove text between <think> and </think> tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def json_to_text(data):
    """Convert structured JSON into a human-readable sentence."""
    text_parts = []
    
    if data.get("Names"):
        text_parts.append(f"The user is named {', '.join(data['Names'])}.")
    
    if data.get("Age"):
        text_parts.append(f"The user is {', '.join(map(str, data['Age']))} years old.")
    
    if data.get("specialization"):
        text_parts.append(f"They specialize in {', '.join(data['specialization'])}.")
    
    if data.get("Topics"):
        text_parts.append(f"They are interested in {', '.join(data['Topics'])}.")
    
    if data.get("Hobbies"):
        text_parts.append(f"Their hobbies include {', '.join(data['Hobbies'])}.")
    
    if data.get("preferred way of learning"):
        text_parts.append(f"They prefer learning through {', '.join(data['preferred way of learning'])}.")
    
    return " ".join(text_parts)


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF to be added to the document retrieval system."""
    file_path = os.path.join(PDF_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    global index
    index = load_pdf()  # Reload index after adding new file
    return {"message": f"Uploaded {file.filename} and updated the document index"}

# @app.post("/extract/")
def extract_information(request: ExtractionRequest):
    """Extract structured information from text using NuExtract."""
    model.to(device)
    model.eval()
    try:
        output = predict_NuExtract(model, tokenizer, request.text, request.schema, request.example)
        return {"extracted_info": json.loads(output)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_NuExtract(model, tokenizer, text, schema, example=["","",""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"

    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to(device)

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]

# @app.post("/chat/")
# async def chat(request: ChatRequest):
#     """Ask a question based on loaded documents."""
#     try:
#         response = chain.run(request.question)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat/")
async def chat(request: ChatRequest):
    """Ask a question based on loaded documents. If it's the first request, prompt the user to introduce themselves."""
    try:
        # If the user is starting a conversation (first request)
        if request.question.strip().lower() in ["", "start", "begin", "hello", "hi"]:
            default_prompt = (
                "Hello! Before we begin, I'd love to get to know you better. "
                "Can you tell me about yourself? I'd like to know:\n"
                "- Your Age\n"
                "- Your Name\n"
                "- Your Specialization\n"
                "- Topics you're interested in\n"
                "- Your Hobbies\n"
                "- Your Preferred Way of Learning"
            )
            return {"response": default_prompt}
        
        extracted_info = predict_NuExtract(model, tokenizer, request.question, schema, example=["", "", ""])

        # extracted_info = extract_information(request.question)
        User_info = json_to_text(json.loads(extracted_info))

        full_prompt =  "The User Information is " + User_info+system_message 


        # full_prompt = ChatPromptTemplate.from_messages([
        #     ("system", system_message + User_info),
        #     ("user", request.question),
        # ])

        # Otherwise, process the user's question normally
        response = chain.run(full_prompt)
        response = remove_think_tags(response)

        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)