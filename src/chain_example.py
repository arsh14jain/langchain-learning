from langchain.prompts import PromptTemplate
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

# Define a prompt template
prompt = PromptTemplate.from_template("Translate the following English text to French: {text}")

# Create a chain using RunnableSequence
chain = prompt | llm

# Run the chain
result = chain.invoke({"text": "Hello, how are you?"})
print(result)