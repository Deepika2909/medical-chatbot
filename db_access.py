from google import genai

# Initialize client
client = genai.Client(api_key="AIzaSyCQzeFE1FQ9UAw0xUOjCvtQ2K1pXakxlNE")

# List models
print("Available models:\n")
for model in client.models.list():
    print("Model:", model.name)

