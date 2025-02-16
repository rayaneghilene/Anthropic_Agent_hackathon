from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
# from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

import json

from glob import glob
import streamlit as st
import ollama
from typing import Dict, Generator

# from create_db import get_or_create_collection, vectorize_text, query_collection, add_paragraphs_to_collection, delete_collection, preprocess_text
import logging
# import chromadb
# from chromadb import Client
# from chromadb.config import Settings
from fpdf import FPDF

logging.getLogger("chromadb.segment.impl.metadata.sqlite").setLevel(logging.ERROR)
logging.getLogger("chromadb.segment.impl.vector.local_hnsw").setLevel(logging.ERROR)


### LLM 🤓
def ollama_generator(messages: Dict) -> Generator:
    stream = ollama.chat(
            model="llava:7b",
            messages=messages,
            stream=True
        )
    for chunk in stream:
        yield chunk['message']['content']



import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Uncomment the following section if you wish to use the Tiny model (0.5B)
model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)

# ## The following model is the 3.7B version
# model = AutoModelForCausalLM.from_pretrained("numind/NuExtract", torch_dtype=torch.bfloat16, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract", trust_remote_code=True)

model.to(device)
model.eval()

folder_path= 'Anthropic_Agent_hackathon/PDF_Documents'
@st.cache_resource
def load_pdf():
    # pdf_name ='Issues with Entailment-based Zero-shot Text Classification.pdf'
    #loaders = [PyPDFLoader(pdf_name)]
    pdf_files = glob(f"{folder_path}/*.pdf")
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]

    index= VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name= 'all-MiniLM-L12-V2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    return index

index = load_pdf()


### Chain

chain= RetrievalQA.from_chain_type(
    llm= ChatOllama(model="deepseek-r1:7b"),
    chain_type ='stuff',
    retriever= index.vectorstore.as_retriever(),
    # retriever= Chroma(client=client, collection_name=collection_name),

    input_key='question'
)



### INTERFACE

with st.sidebar:
    st.title('Anthropic Agent Hackathon')
    st.image('/Users/rayaneghilene/Documents/Anthropic_Hackathon/Anthropic_Agent_hackathon/Images/image.png',  use_column_width='auto')
    # st.file_uploader('Upload your own file')
st.title('Anthropic Agent Hackathon')

if 'messages' not in st.session_state:
    st.session_state.messages = []
    first_message = "Hi there!Can you tell me a bit about yourself? I'd love to know your name, age, your area of specialization, and the topics you're most interested in. Also, what are some of your hobbies, and how do you prefer to learn new things (e.g., hands-on practice, reading, watching videos, or discussions)"
    st.chat_message("assistant").markdown(first_message)

    st.session_state.messages.append( {"role": "assistant", "content": first_message})   

    

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Insert your text here :)')

schema = """{ 
    "Age": [],
    "Names": [],
    "specialization": [],
    "Topics": [],
    "Hobbies": [],
    "preffered way of learning": []
}"""

system_message = """You are an AI assistant designed to collect specific information from the user in a structured and conversational manner. Your goal is to extract the following details:
	•	Age: Ask for the user's age or age range. If they are uncomfortable sharing an exact number, accept a general range.
	•	Names: Request their name or preferred name. Allow for multiple names if necessary.
	•	Specialization: Ask about their field of expertise, academic background, or professional focus.
	•	Topics of Interest: Inquire about subjects they are currently interested in, studying, or working on.
	•	Hobbies: Gather information about their leisure activities and interests outside of work/study.
	•	Preferred Way of Learning: Ask how they best absorb new information—through reading, videos, hands-on practice, structured courses, etc. 
Maintain a friendly and engaging tone, adapt to the user's responses dynamically, and encourage them to provide details without making the conversation feel like a survey. If the user provides incomplete answers, gently prompt them for more details."""

# if prompt: 
#     st.chat_message('user').markdown(prompt)
#     st.session_state.messages.append({'role': 'user', 'content': prompt})
    
#     st.spinner(text='In progress')
#     model.to(device)
#     model.eval()
#     new_prediction = predict_NuExtract(model, tokenizer, prompt, schema, example=["","",""])
#     full_prompt = ChatPromptTemplate.from_messages([
#         ("system", system_message),
#         ("user", prompt),
#     ])
#     response = chain.run(full_prompt.format())
#     # response = chain.run(prompt)
#     st.success('Done!')
#     st.chat_message("assistant").markdown(response)

#     st.session_state.messages.append( {"role": "assistant", "content": response})   


import streamlit as st
from langchain.prompts import ChatPromptTemplate
import json

# Define schema
schema = """{
    "Age": [],
    "Names": [],
    "specialization": [],
    "Topics": [],
    "Hobbies": [],
    "preferred way of learning": []
}"""

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


# Ensure session state has a place to store predictions
# if "predictions" not in st.session_state:
#     st.session_state.predictions = schema  # Initialize with the schema structure

system_message = "You are an AI assistant that extracts structured information from text, based on the Json Schema provided generate a prompt for an LLM to perform an action."

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    st.spinner(text='In progress')

    model.to(device)
    model.eval()

    # Run the prediction
    new_prediction = predict_NuExtract(model, tokenizer, prompt, schema, example=["", "", ""])
    User_info = json_to_text(json.loads(new_prediction))
    # Append the new prediction to the existing session state
    # for key in new_prediction:
    #     if key in st.session_state.predictions:
    #         st.session_state.predictions[key].extend(new_prediction[key])

    # Define the full prompt including the system message
    full_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message + User_info),
        ("user", prompt),
    ])

    # Run LangChain Model
    response = chain.run(full_prompt.format())
    response = remove_think_tags(response)
    mapped_response =f"""the system response is: {response}"""
    st.success('Done!')
    st.chat_message("assistant").markdown(mapped_response)

    st.session_state.messages.append({"role": "assistant", "content": mapped_response})

    # Display updated schema with appended predictions
    # st.json(st.session_state.predictions)
