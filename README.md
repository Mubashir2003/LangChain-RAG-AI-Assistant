🤖 Real-Time AI Assistant with RAG & LangChain

This project builds a local, real-time AI assistant capable of searching the internet to answer current questions. It leverages Ollama for local inference and DuckDuckGo for privacy-focused web searches, orchestrated by LangChain.

🛠️ Environment Setup

Before running the code, ensure your local environment is ready to host the Large Language Model (LLM).

Install Ollama:
Download the application for your operating system (Windows, Mac, or Linux) at ollama.com.

Pull the Model:
Open your terminal or command prompt and download the Llama 3 model (8 billion parameters):

ollama pull llama3:8b


📦 Install Dependencies

Install the necessary Python libraries to connect LangChain, Ollama, and the search tools.

pip install langchain langchain-community langchain-ollama duckduckgo-search


Note: If you encounter version conflicts with duckduckgo-search, try pinning the version: pip install duckduckgo-search==5.3.1

🧠 Architecture & Logic

This assistant uses Retrieval Augmented Generation (RAG). Instead of relying solely on the AI's training data (which can be outdated), we inject live information into the conversation.

The Brain (LLM): llama3:8b running locally via Ollama.

The Eyes (Tool): DuckDuckGoSearchRun fetches live data from the web.

The Orchestrator (LCEL): LangChain Expression Language connects the user input ➡️ search tool ➡️ prompt ➡️ LLM.

💻 The Source Code

Save the following code in a file named assistant.py.

from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Initialize the Model
# We specify the model we pulled earlier via Ollama
llm = Ollama(model="llama3:8b")

# 2. Initialize the Search Tool
# This tool will run a web search using DuckDuckGo
search = DuckDuckGoSearchRun()

# 3. Define the Prompt Template
# This is the instruction manual for the LLM. 
# It strictly instructs the AI to use the provided search context.
prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. You must answer the user's question 
    based *only* on the following search results. If the search results 
    are empty or do not contain the answer, say 'I could not find 
    any information on that.'

    Search Results:
    {context}

    Question:
    {question}
    """
)

# 4. Build the RAG Chain using LCEL
chain = (
    RunnablePassthrough.assign(
        # "context" is dynamically filled by running the search tool
        # with the user's question.
        context=lambda x: search.run(x["question"])
    )
    | prompt  # Pipe the context + question into the prompt
    | llm     # Pipe the formatted prompt into Llama 3
)

# 5. Execution Loop
print("🤖 Hello! I'm a real-time AI assistant. What's new?")

while True:
    try:
        user_query = input("You: ")
        
        # Exit condition
        if user_query.lower() in ["exit", "quit"]:
            print("🤖 Goodbye!")
            break
        
        print("🤖 Thinking...")
        
        # Invoke the chain with the user's question
        response = chain.invoke({"question": user_query})
        
        print(f"🤖: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")


🚀 How to Run

Open your terminal in VS Code.

Run the script:

python AI-assistant.py


Ask a question about current events (e.g., "What is the current price of Bitcoin?") to see the search tool in action!
