import os
import pandas as pd
from dotenv import load_dotenv
#from langchain_community.llms import Ollama
#from langchain_community.chat_models import ChatOllama
# In src/agent.py
from langchain_ollama import ChatOllama
# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import tool, AgentExecutor, create_react_agent
# LangChain integrations for Google Gemini and HuggingFace Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
# --- 1. Load API Key ---
load_dotenv()
# --- 2. Language Mapping ---
LANGUAGE_MAP = {
    "6": "Hindi", "7": "Kannada", "11": "Malayalam",
    "20": "Tamil", "21": "Telugu", "24": "English"
}
# --- 3. Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    # convert_system_message_to_human=True # Uncomment if you face system prompt issues
)
#llm = Ollama(
#    model="llama3.1:8b"
#)
#llm = ChatOllama(
#    model="llama3.1:8b"
#)
# --- 4. Data Loading and RAG Setup ---
def create_vector_db():
    """Loads data from the CSV, processes it, and creates a FAISS vector store."""
    try:
        df = pd.read_csv('data/courses.csv')
    except FileNotFoundError:
        print("Error: data/courses.csv not found. Please ensure the file is in the correct location.")
        return None
    # --- Data Preprocessing ---
    # Map language codes to full names for better search context
    df['Course Released Languages'] = df['Course Released Languages'].apply(
        lambda x: ', '.join([LANGUAGE_MAP.get(code.strip(), 'Unknown') for code in str(x).split(',')])
    )
    # Combine relevant columns into a single text document for each course
    df['combined_text'] = (
        "Course Title: " + df['Course Title'].astype(str) + "; " +
        "About Course: " + df['About Course'].astype(str) + "; " +
        "Target Audience: " + df['Who This Course Is For'].astype(str) + "; " +
        "Available Languages: " + df['Course Released Languages'].astype(str)
    )
    loader = DataFrameLoader(df, page_content_column='combined_text')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    # Use a standard embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and return the vector store
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db
# Create the vector database instance. This will be done once when the app starts.
print("Creating vector database...")
VECTOR_DB = create_vector_db()
print("Vector database created successfully.")
# --- 5. Define Tools for the Agent ---
@tool
def course_search(query: str) -> str:
    """
    Searches the Boss Wallah course database to answer questions about courses.
    Use this for any questions about course content, descriptions, languages, or target audience.
    """
    if VECTOR_DB is None:
        return "Sorry, the course database is not available."
        
    retriever = VECTOR_DB.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create a prompt for the RAG chain
    prompt_template = """
    You are a factual data extractor. Your ONLY job is to analyze the CONTEXT provided and find information that directly answers the USER'S QUERY.
    **RULES:**
    1.  Base your entire answer *only* on the text provided in the CONTEXT.
    2.  Do not make up any courses, details, or information.
    3.  If the CONTEXT does not contain any relevant courses that match the query, you MUST reply with the exact phrase: "No relevant courses found."
    4.  If you find relevant courses, summarize them concisely.
    CONTEXT:
    {context}
    USER'S QUERY:
    {query}
    YOUR ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Create a chain to process the query with the retrieved context
    rag_chain = prompt | llm
    response = rag_chain.invoke({"context": context, "query": query})
    return response
@tool
def web_search(query: str) -> str:
    """
    Use this tool for questions that are NOT about Boss Wallah's courses.
    This includes general knowledge, current events, or finding external information like store locations.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)
# --- 6. Create the Agent Executor ---
def create_agent_executor():
    """Creates and returns the LangChain agent that can use the defined tools."""
    tools = [course_search, web_search]
    
    # This prompt is the agent's "brain", guiding its decisions
    # This prompt is the agent's "brain", guiding its decisions
    # This prompt is the agent's "brain", guiding its decisions
    # FINAL, POLISHED PROMPT - This version contains all required variables.
    agent_prompt_template = """
    You are "Boss Bot," an expert AI assistant for Boss Wallah. Your primary goal is to help users find the perfect course.
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    **Your Decision-Making Process (Follow these rules strictly):**
    1.  **Prioritize Course Search:** Your default tool is ALWAYS `course_search`. Your main purpose is to find relevant courses from the database.
    2.  **Analyze User Intent:**
        * If the user asks about a **specific topic** (e.g., "poultry farming," "agribusiness"), you MUST use `course_search`.
        * If the user **describes themselves** or their situation (e.g., "I am a high school graduate," "I'm a beginner looking to start a business"), you MUST use `course_search` to find courses that match their profile in the "Who This Course Is For" section.
        * Only use `web_search` if the question is **clearly external** and cannot possibly be answered by a course description (e.g., "Where are stores in Bangalore?", "What is the weather today?").
    ---
    **Example of a Complete Interaction:**
    ---
    Question: Where can I buy seeds in Bangalore?
    Thought: The user is asking for a location. According to Rule #2, this is a clearly external question, so I must use the web_search tool.
    Action: web_search
    Action Input: "buy seeds in Bangalore"
    Observation: A web search returns several nurseries and agricultural stores in Bangalore.
    Thought: I have the information from the web search and can now formulate the final answer.
    Final Answer: You can buy seeds at several agricultural stores in Bangalore, such as the one at Lalbagh Botanical Garden.
    ---
    3.  **Handle Ambiguity:** If a user's question is too vague (e.g., "tell me more," "what about those"), you must ask for clarification instead of using a tool.
    Use the following format:
    Question: the input question you must answer
    Thought: I will follow my decision-making process. First, I will analyze the user's intent. Based on the rules, I will select the best tool.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    Thought: I now have the final answer.
    Final Answer: the final answer to the original input question
    
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    4.  **Complete the Task:** After you receive an "Observation" from a tool, you MUST continue the process. Your next step is to write a final "Thought" and then provide the "Final Answer:". Do not stop until you have provided the Final Answer.
    
    """
    
    
    prompt = PromptTemplate.from_template(agent_prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=3)
    
    return agent_executor