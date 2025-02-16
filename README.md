# Anthropic_Agent_hackathon

![Anthropic agent hackathon image](https://github.com/rayaneghilene/Anthropic_Agent_hackathon/blob/main/Images/image.png)

## usage
To clone the repository run: 

```bash
git clone https://github.com/rayaneghilene/Anthropic_Agent_hackathon.git
cd Anthropic_Agent_hackathon
```


Create avirtual environment (recommended ):
```bash
python -m venv hackathon
source hackathon/bin/activate
```



Install the requirements:

```ruby
pip install -r requirements.txt
```


Install Ollama:

You need to install **Ollama** locally on your machine to run this code. [Link to install ollama](https://ollama.com/) 

Once installed you need to import the Llava:7b model. You can do so using the following command on the terminal:

```bash
ollama pull deepseek-r1:7b
```
and 
```bash
ollama run deepseek-r1:7b
```

You can also load a different model from the ollama library. Check out the available models [here]( https://ollama.com/library)




## Run the streamlit interface:
```ruby
streamlit run app.py
```

## Run the API 
```ruby
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

You can quickly test the API chat with the test_api.py script

```bash
python test_api.py
```
