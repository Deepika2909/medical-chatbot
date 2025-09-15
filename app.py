import streamlit as st
from rag_pipeline import MedicalRAGChatbot

# -----------------------------
# CONFIGURATION (hardcoded)
# -----------------------------
GEMINI_API_KEY = "ADD API HERE"  # <-- Add your Gemini API key here
CSV_FILE_PATH = "data/medical.csv"           # <-- Add the path to your CSV file here

# Initialize chatbot once
if "chatbot" not in st.session_state:
    with st.spinner("🔄 Initializing chatbot... This may take a few minutes for first-time setup."):
        st.session_state.chatbot = MedicalRAGChatbot(CSV_FILE_PATH, GEMINI_API_KEY)

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Medical FAQ Chatbot",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .subtitle { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 1rem 0; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .assistant-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE AND DESCRIPTION
# -----------------------------
st.markdown('<h1 class="main-header">🏥 Medical FAQ Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask me any medical question and I\'ll search through thousands of medical FAQs to give you accurate answers!</p>', unsafe_allow_html=True)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# WELCOME MESSAGE & SAMPLE QUESTIONS
# -----------------------------
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try asking questions like:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **🩺 Symptoms & Diagnosis**
        - What are the early symptoms of diabetes?
        - How is hypertension diagnosed?
        - What causes chest pain?
        - Signs of heart disease?
        """)
    with col2:
        st.info("""
        **💊 Treatment & Medication**
        - Can children take paracetamol?
        - What foods are good for heart health?
        - How to treat a common cold?
        - Natural remedies for headaches?
        """)

    st.markdown("### 🎯 Quick Start - Click a sample question:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What are diabetes symptoms?"):
            st.session_state.sample_query = "What are the early symptoms of diabetes?"
    with col2:
        if st.button("Heart healthy foods?"):
            st.session_state.sample_query = "What foods are good for heart health?"
    with col3:
        if st.button("Can kids take paracetamol?"):
            st.session_state.sample_query = "Can children take paracetamol safely?"

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View Sources ({len(message['sources'])} documents)", expanded=False):
                for i, (qtype, question, answer, score) in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** *(Type: {qtype}, Relevance: {score:.3f})*")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer[:300]}..." if len(answer) > 300 else f"**A:** {answer}")
                    if i < len(message["sources"]) - 1:
                        st.divider()

# -----------------------------
# HANDLE USER INPUT
# -----------------------------
if "sample_query" in st.session_state:
    prompt = st.session_state.sample_query
    del st.session_state.sample_query
else:
    prompt = st.chat_input("💬 Ask your medical question here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching medical knowledge base..."):
            response = st.session_state.chatbot.chat(prompt)
        st.markdown(response['answer'])
        
        if response['sources_used'] > 0:
            with st.expander(f"📚 View Sources ({response['sources_used']} documents)", expanded=False):
                for i, (qtype, question, answer, score) in enumerate(response['relevant_docs']):
                    st.markdown(f"**Source {i+1}** *(Type: {qtype}, Relevance: {score:.3f})*")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer[:300]}..." if len(answer) > 300 else f"**A:** {answer}")
                    if i < len(response['relevant_docs']) - 1:
                        st.divider()
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response['answer'],
        "sources": response['relevant_docs']
    })

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This chatbot is for informational purposes only. 
        The information provided is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Statistics")
    total_faqs = len(st.session_state.chatbot.df)
    st.metric("Total FAQs", f"{total_faqs:,}")
    st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))

# -----------------------------
# CLEAR CHAT HISTORY
# -----------------------------
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
