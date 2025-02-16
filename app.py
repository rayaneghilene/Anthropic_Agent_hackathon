from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
# from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List, Optional
import urllib.parse

import json
import anthropic
from glob import glob
import streamlit as st
import ollama
import requests
from typing import Dict, Generator

import logging
from fpdf import FPDF

logging.getLogger("chromadb.segment.impl.metadata.sqlite").setLevel(logging.ERROR)
logging.getLogger("chromadb.segment.impl.vector.local_hnsw").setLevel(logging.ERROR)


class BraveAPI:
    def __init__(self, api_key: str):
        """Initialize Brave Search API client."""
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1"
        # self.base_url = "https://api.search.brave.com"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
    
    def search(self, query: str, count: int = 5) -> List[Dict]:
        """
        Perform a web search using Brave Search API.
        
        Args:
            query: Search query string
            count: Number of results to return
            
        Returns:
            List of search results with title, description, and URL
        """
        try:
            # Clean and format the query
            cleaned_query = self._clean_search_query(query)
            print(f"Requesting: {self.base_url}")
            print(f"Headers: {self.headers}")
            # print(f"Params: {params}")
            response = requests.get(
                f"{self.base_url}/web/search/",
                headers=self.headers,
                params={
                    "q":cleaned_query,
                    "count": count
                }
            )
            response.raise_for_status()
            # data = response.json()
            if response.status_code == 200:
                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError:
                    logging.error("Failed to decode JSON response.")
                    print(response.text)  # Debug: Print the response content
                    return []  # Return empty list or handle accordingly
            else:
                logging.error(f"Brave API returned {response.status_code}: {response.text}")
                return []  # Handle error properly
            
            results = []
            for web in data.get("web", {}).get("results", []):
                results.append({
                    "title": web.get("title"),
                    "description": web.get("description"),
                    "url": web.get("url")
                })
            return results
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Brave Search API error: {str(e)}")
            raise

    
    def _clean_search_query(self, query: str) -> str:
        """Clean and format the search query for Brave Web Search API."""
        # Normalize whitespace and remove leading/trailing spaces
        cleaned = ' '.join(query.strip().split())
        # Remove excessive special characters while keeping meaningful punctuation
        cleaned = re.sub(r"[^\w\s.,!?'-]", '', cleaned)
        # Limit query length to 500 characters
        cleaned = cleaned[:500]
        # URL-encode the query for safe transmission
        return urllib.parse.quote_plus(cleaned)
class EnhancedClaudeAPI:
    def __init__(self, claude_api_key: str, brave_api_key: str):
        """Initialize enhanced Claude API with Brave Search capabilities."""
        self.claude = anthropic.Anthropic(api_key=claude_api_key)
        self.brave = BraveAPI(brave_api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def search_and_generate(self, 
                           user_query: str,
                           system_message: str,
                           search_results_count: int = 3,
                           max_tokens: int = 1024,
                           temperature: float = 0.7) -> str:
        """
        Search for relevant information and generate a response using Claude.
        
        Args:
            user_query: User's question or prompt
            system_message: System message for Claude
            search_results_count: Number of search results to include
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-1)
            
        Returns:
            Claude's response incorporating search results
        """
        try:
            # Perform web search
            search_results = self.brave.search(user_query, search_results_count)
            
            # Format search results for Claude
            search_context = "Here are some relevant search results:\n\n"
            for i, result in enumerate(search_results, 1):
                search_context += f"{i}. {result['title']}\n"
                search_context += f"   Description: {result['description']}\n"
                search_context += f"   URL: {result['url']}\n\n"
            
            # Combine search results with user query
            enhanced_prompt = f"""Search results:
            {search_context}
            User query: {user_query}
            Please provide a response incorporating relevant information from the search results."""

            # Generate response using Claude
            messages = [
                # {"role": "system", "content": system_message},
                {"role": "user", "content": system_message + enhanced_prompt}
            ]
            
            response = self.claude.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            
            return response.content
            
        except Exception as e:
            logging.error(f"Error in search and generate: {str(e)}")
            raise


def init_enhanced_claude():
    """Initialize Enhanced Claude API with Brave Search integration."""
    claude_key = st.secrets.get("CLAUDE_API_KEY")
    brave_key = st.secrets.get("BRAVE_API_KEY")
    
    if not claude_key or not brave_key:
        st.error("Missing API keys in secrets. Please add both CLAUDE_API_KEY and BRAVE_API_KEY.")
        return None
    
    try:
        enhanced_claude = EnhancedClaudeAPI(claude_key, brave_key)
        return enhanced_claude
    except Exception as e:
        st.error(f"Error initializing Enhanced Claude API: {str(e)}")
        return None
    

if 'enhanced_claude' not in st.session_state:
    st.session_state.enhanced_claude = init_enhanced_claude()


def process_user_input_with_search(user_input: str, system_message: str) -> str:
    """Process user input with web search and Claude API."""
    if not st.session_state.enhanced_claude:
        return "Enhanced Claude API not properly initialized. Please check your API keys."
        
    try:
        response = st.session_state.enhanced_claude.search_and_generate(
            user_input,
            system_message
        )
        return response
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return f"Error processing request: {str(e)}"



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

model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)

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

    

# for message in st.session_state.messages:
#     st.chat_message(message['role'])#.markdown(message['content'])

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

# def remove_think_tags(text):
#     """Remove text between <think> and </think> tags."""
#     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def remove_think_tags(text):
    if isinstance(text, list):  # If it's a list, join elements into a string
        text = " ".join(map(str, text))
    elif not isinstance(text, str):  # If it's not a string, convert to string
        text = str(text)
    
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


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= CLAUDE_API_KEY ,
)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)




system_message = "You are an AI assistant that extracts structured information from text, based on the Json Schema provided generate a prompt for an ai agent to perform an action, the prompt should guide the agent to search the relevant informations form the internet and return useful study tools, like QCM"

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    st.spinner(text='In progress')

    model.to(device)
    model.eval()

    # Run the prediction
    userinfo = predict_NuExtract(model, tokenizer, prompt, schema, example=["", "", ""])
    User_info = json_to_text(json.loads(userinfo))

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
    search_system_message = "based on the prompt perform a websearch and return the relevant informations using the feinman studyin technique, you should help the user understand the topic effeortlessly and provide different types of study tools like QCM, videos, articles, etc"
    agent_response = st.session_state.enhanced_claude.search_and_generate(
                prompt,
                userinfo,
                search_system_message
            )
    if agent_response:
                mapped_response = f"System response (with web search results): {remove_think_tags(agent_response)}"
                st.chat_message("assistant").markdown(mapped_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": mapped_response
                })
    # agent_response = process_user_input_with_search( mapped_response, search_system_message)

    st.chat_message("assistant")
    # .markdown(agent_response)

    st.session_state.messages.append({"role": "assistant", "content": agent_response})

