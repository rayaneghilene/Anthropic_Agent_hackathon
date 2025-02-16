# Anthropic_Agent_hackathon
We use ollama to run the deepseek r1 qwen 7B model for master prompt generation. This includes understanding the user's level of experience and relevant informations (hobbies, interests) which can be useful when generating tailored responses. The Data colloected from the user us processed using the NUmind Nu Extract 0.5B NER model, to extract the relevant informations, and formatt it as a prompt to query the Deepseek model. The deepseek model returns a master prompt that us fed to the Claude AI Agent, which uses to the Brave API to perform searches ang generats a comprehensie studying guide with respect to the feynman method. 

![Anthropic agent hackathon image](https://github.com/rayaneghilene/Anthropic_Agent_hackathon/blob/main/Images/image.png)

## Preview of the Agent
https://github.com/rayaneghilene/Anthropic_Agent_hackathon/blob/main/Images/Preview.mov

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





## Contributing
We welcome contributions from the community to enhance work. If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr
