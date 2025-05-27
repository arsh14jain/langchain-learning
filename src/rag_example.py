from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('GOOGLE_API_KEY')

# client = genai.Client(api_key=api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a vector store
texts = ["LangChain is a framework for developing applications powered by language models - this is langchain rag example"]
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# Create a retriever
retriever = vectorstore.as_retriever()

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
answer = qa_chain.invoke({"query": "What is LangChain?"})
print(answer)