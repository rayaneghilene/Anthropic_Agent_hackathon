import requests

url = "http://127.0.0.1:8000/chat/"
data = {
    "question": "Hi my name is Rayan. I'm 23 and specialize in neuroscience. I'm into running and swimming. I want to learn about sensory systems and the brain, and my preferred way of learning is through diagrams and videos."
}

response = requests.post(url, json=data)
# print(response.json())  # Prints the server's response


if response.status_code == 200:
    print("Chatbot Response:", response.json()["response"])
else:
    print("Error:", response.json())