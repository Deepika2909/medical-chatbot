# Medical FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a knowledge base of medical FAQs.

## Features

- **Smart Search**: Uses semantic similarity to find relevant medical information
- **AI-Powered Answers**: Generates natural, contextual responses using OpenAI GPT
- **User-Friendly Interface**: Clean Streamlit web app
- **Source Transparency**: Shows which documents were used to generate answers

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key (get from https://platform.openai.com/api-keys)
- Your medical FAQ CSV file with columns: `qtype`, `Question`, `Answer`

### 2. Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running the Chatbot

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and go to the URL shown (usually `http://localhost:8501`)

3. In the sidebar:
   - Enter your OpenAI API key
   - Enter the path to your CSV file
   - Click "Initialize Chatbot"

4. Start asking medical questions!

### 4. First Run Notes

- **First time setup takes 2-5 minutes** to create embeddings for your dataset
- Embeddings are saved in the `embeddings/` folder for faster future runs
- The chatbot works best with specific medical questions

## How It Works

### RAG Pipeline

1. **Data Loading**: Reads medical FAQ CSV file
2. **Embedding Creation**: Converts all Q&A pairs to vector embeddings
3. **Vector Storage**: Stores embeddings in FAISS index for fast similarity search
4. **Retrieval**: Finds most relevant documents for user queries
5. **Generation**: Uses OpenAI GPT to generate natural answers based on retrieved context

### Technology Stack

- **Streamlit**: Web interface
- **Sentence Transformers**: Text embeddings (`all-MiniLM-L6-v2` model)
- **FAISS**: Vector similarity search
- **OpenAI GPT-3.5-turbo**: Answer generation
- **Pandas**: Data manipulation

## Example Queries

- "What are the early symptoms of diabetes?"
- "Can children take paracetamol?"
- "What foods are good for heart health?"
- "How is hypertension diagnosed?"
- "What causes chest pain?"

## Project Structure

```
medical_chatbot/
├── app.py                 # Streamlit interface
├── rag_pipeline.py        # RAG system core logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── embeddings/           # Generated embeddings (created on first run)
└── your_data.csv         # Your medical FAQ dataset
```

## Design Choices

### 1. **Embedding Model**: `all-MiniLM-L6-v2`
- Fast and efficient
- Good balance of speed and quality
- Works well for medical text

### 2. **Vector Database**: FAISS
- Free and fast
- Excellent for similarity search
- Easy to set up locally

### 3. **LLM**: OpenAI GPT-3.5-turbo
- Reliable medical knowledge
- Good at following context
- Cost-effective

### 4. **Retrieval Strategy**
- Combines questions and answers for richer context
- Uses cosine similarity for semantic matching
- Returns top 3 most relevant documents

## Troubleshooting

### Common Issues

1. **"No module named 'rag_pipeline'"**
   - Make sure both `app.py` and `rag_pipeline.py` are in the same directory

2. **OpenAI API errors**
   - Check your API key is valid and has credits
   - Verify internet connection

3. **CSV loading errors**
   - Ensure your CSV has columns: `qtype`, `Question`, `Answer`
   - Check file path is correct

4. **Slow first run**
   - First run creates embeddings for 16k+ documents
   - Subsequent runs are much faster

## Performance

- **Dataset Size**: 16,407 medical FAQs
- **First Run**: 2-5 minutes (creating embeddings)
- **Subsequent Runs**: < 30 seconds to start
- **Query Response**: 3-10 seconds per question

## Disclaimer

This chatbot is for informational purposes only. Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.