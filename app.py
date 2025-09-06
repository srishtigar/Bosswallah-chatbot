import streamlit as st
from src.agent import create_agent_executor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- App Configuration ---
st.set_page_config(
    page_title="Boss Wallah AI Chatbot",
    page_icon="🏢",
    layout="wide"
)

# --- Custom CSS for Professional Boss Wallah Branding ---
# Clean design: Orange header, white background, black text, left-aligned chat
page_style = """
<style>
/* Import professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Remove default Streamlit styling */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main app background - clean white */
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    font-family: 'Inter', sans-serif;
}

/* Header styling - orange like BossWallah */
[data-testid="stHeader"] {
    background-color: #FF6B35;
    height: 80px;
}

/* Custom header section */
.boss-wallah-header {
    background-color: #FF6B35;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.boss-wallah-logo {
    color: #FFFFFF;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.boss-wallah-tagline {
    color: #FFFFFF;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 2px;
    opacity: 0.9;
}

/* Main content area - white background */
.main .block-container {
    background-color: #FFFFFF;
    padding-top: 0;
    max-width: 1000px;
}

/* Main title - Large heading */
h1 {
    color: #2C2C2C;
    font-weight: 600;
    font-size: 2.5rem;
    text-align: center;
    margin: 2rem 0 1rem 0;
    font-family: 'Inter', sans-serif;
    line-height: 1.2;
}

/* Description text - Medium size, left aligned */
.stMarkdown p {
    color: #4A4A4A;
    font-size: 1.1rem;
    line-height: 1.6;
    text-align: left;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* --- FIX FOR AVATAR ALIGNMENT --- */
/* Chat messages styling - LEFT ALIGNED */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    border: 1px solid #E8E8E8;
    background-color: #FFFFFF;
    font-size: 1rem;
    line-height: 1.5;
    text-align: left !important;
    display: flex; /* Changed from 'block' to 'flex' */
    align-items: center; /* Vertically centers the avatar and text */
    gap: 1rem; /* Adds space between avatar and text */
}
/* --- END FIX --- */


/* Assistant messages - subtle orange background, LEFT ALIGNED */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFF8F5;
    border-left: 4px solid #FF6B35;
    border-top: 1px solid #FFE5D9;
    border-right: 1px solid #FFE5D9;
    border-bottom: 1px solid #FFE5D9;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
    color: #2C2C2C !important;
    font-weight: 400;
    font-size: 1rem;
    text-align: left !important;
    margin: 0;
}

/* User messages - clean white with subtle border, LEFT ALIGNED */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #FAFAFA;
    border: 1px solid #E0E0E0;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
    color: #2C2C2C !important;
    font-weight: 400;
    font-size: 1rem;
    text-align: left !important;
    margin: 0;
}

/* Chat input styling */
[data-testid="stChatInput"] {
    border-radius: 10px;
    border: 2px solid #FF6B35;
    background-color: #FFFFFF;
}

[data-testid="stChatInput"] input {
    border: none;
    background-color: transparent;
    color: #2C2C2C;
    font-size: 1rem;
    font-weight: 400;
}

[data-testid="stChatInput"] input::placeholder {
    color: #888888;
    font-weight: 400;
}

/* Chat input button */
[data-testid="stChatInput"] button {
    background-color: #FF6B35;
    border: none;
    border-radius: 8px;
    color: #FFFFFF;
}

[data-testid="stChatInput"] button:hover {
    background-color: #E55A2E;
}

/* Avatar styling - professional */
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #FF6B35 !important;
    color: #FFFFFF !important;
    font-size: 1.2rem;
}

[data-testid="chatAvatarIcon-user"] {
    background-color: #4A4A4A !important;
    color: #FFFFFF !important;
    font-size: 1.2rem;
}

/* Info box styling - professional, LEFT ALIGNED */
[data-testid="stInfo"] {
    background-color: #F8F9FA;
    border: 1px solid #DEE2E6;
    border-left: 4px solid #FF6B35;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    text-align: left !important;
}

[data-testid="stInfo"] p {
    color: #495057 !important;
    font-weight: 400;
    font-size: 0.95rem;
    margin: 0;
    text-align: left !important;
}

/* Spinner color */
[data-testid="stSpinner"] > div {
    border-top-color: #FF6B35;
}

/* Success and error messages - professional */
[data-testid="stSuccess"] {
    background-color: #F8F9FA;
    border: 1px solid #28a745;
    border-left: 4px solid #28a745;
    border-radius: 8px;
    color: #2C2C2C;
}

[data-testid="stError"] {
    background-color: #FDF2F2;
    border: 1px solid #dc3545;
    border-left: 4px solid #dc3545;
    border-radius: 8px;
    color: #2C2C2C;
}

/* Footer styling */
.footer {
    text-align: center;
    color: #888888;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #E8E8E8;
    font-weight: 400;
}

/* Force left alignment for all text content */
.stMarkdown, .stChatMessage, [data-testid="stChatMessage"] {
    text-align: left !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .boss-wallah-header {
        padding: 1rem;
        margin: -1rem -1rem 1rem -1rem;
    }
    
    .boss-wallah-logo {
        font-size: 1.5rem;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .stMarkdown p {
        font-size: 1rem;
    }
}

/* Remove excessive spacing */
.element-container {
    margin-bottom: 0.5rem;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- BossWallah Header ---
st.markdown("""
<div class="boss-wallah-header">
    <div class="boss-wallah-logo">
        🏢 BOSS WALLAH
    </div>
    <div class="boss-wallah-tagline">BE THE BOSS</div>
</div>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("Boss Wallah AI Support Chatbot")
st.markdown(
    """
    I can help you with Boss Wallah courses, expert connections, and business information.
    """
)
st.info("Try asking: 'Tell me about the financial freedom course' or 'Where can I buy seeds in Whitefield, Bangalore?'")

# --- Initialize the Agent ---
if 'agent_executor' not in st.session_state:
    with st.spinner("Initializing AI Agent..."):
        st.session_state.agent_executor = create_agent_executor()

# --- Chat History Management ---
if 'memory' not in st.session_state:
    st.session_state.memory = ChatMessageHistory()
    # Add the initial greeting to memory - SHORTER MESSAGE
    st.session_state.memory.add_ai_message("Hello! How can I assist you today?")

# Display messages from memory
for message in st.session_state.memory.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- User Input and Chat Logic ---
if prompt := st.chat_input("Ask your business question here..."):
    # Add user message to memory and display it
    st.session_state.memory.add_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                # Invoke the agent with memory for conversation history
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.memory.messages
                })
                # Add AI response to memory and display it
                st.session_state.memory.add_ai_message(response['output'])
                st.markdown(response['output'])
            except Exception as e:
                error_message = f"I apologize, but I encountered an issue: {e}. Please try rephrasing your question or try again."
                st.error(error_message)

# --- Footer ---
st.markdown("""
<div class="footer">
    Powered by Boss Wallah Technologies | Be the Boss
</div>
""", unsafe_allow_html=True)