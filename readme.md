# 🤖 Boss Wallah - AI Support Chatbot

A comprehensive AI chatbot built for the Boss Wallah assignment. It uses a Retrieval-Augmented Generation (RAG) architecture with an agentic framework to provide accurate answers about courses and also search the web for external information.

## ✨ Features

- **Dual-Capability Agent**: Intelligently switches between searching a local course database (RAG) and performing a live web search.
- **High-Fidelity Answers**: Uses FAISS vector search to provide accurate, context-aware answers based on the provided course dataset.
- **Powered by Gemini**: Leverages Google's gemini-2.5-flash model for fast and effective reasoning and response generation.
- **Interactive UI**: A clean, user-friendly chat interface built with Streamlit.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 2.5 Flash |
| Frameworks | LangChain, Streamlit |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Agent Type | ReAct (Reasoning and Acting) |

## 📂 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `data/courses.csv` | The dataset containing all course information. |
| `src/agent.py` | Core backend logic for the RAG pipeline and agent creation. |
| `app.py` | The main Streamlit application file for the UI. |
| `.env` | Stores the GOOGLE_API_KEY for authentication. |
| `requirements.txt` | A list of all Python dependencies required for the project. |

## ⚙️ Technical Implementation

The project's core is a LangChain ReAct Agent that has access to two custom tools. The agent's prompt guides it to prioritize the course_search tool for relevant queries and fall back to web_search for external questions.

### Agent Architecture (src/agent.py)

```python
# 1. Define Tools for the Agent
@tool
def course_search(query: str) -> str:
    """Searches the Boss Wallah course vector database."""
    retriever = VECTOR_DB.as_retriever()
    # ... RAG chain logic ...
    return response

@tool
def web_search(query: str) -> str:
    """Searches the web for external information."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# 2. Create the Agent with Tools and a Custom Prompt
tools = [course_search, web_search]

agent_prompt = PromptTemplate.from_template(...) # Custom instructions for the agent

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

## 🚀 Setup and Installation
    ```
1.  **Create a `.env` file:**
    Create a file named `.env` in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

## Future Plans

- **User Logins**: Add logins so students and instructors can have their own personalized experience
- **Learning Dashboard**: Create a dashboard with charts to visually track learning progress and see popular courses
- **Smarter Chatbot**: Improve the AI so it can remember conversations and automatically suggest the best courses for each user

## 📸 Screenshots

