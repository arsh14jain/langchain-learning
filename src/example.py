from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('GOOGLE_API_KEY')

# client = genai.Client(api_key=api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

response = llm.invoke("Explain how AI works in a few words")

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[{"parts": [{"text": "Explain how AI works in a few words"}]}]
# )

print(response.text)
