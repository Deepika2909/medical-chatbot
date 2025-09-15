import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pickle
import os
from typing import List, Tuple

class MedicalRAGChatbot:
    def __init__(self, csv_path: str, GEMINI_API_KEY: str):
        self.csv_path = csv_path
        self.GEMINI_API_KEY = GEMINI_API_KEY

        # Initialize the Gemini client with the API key
        genai.configure(api_key=self.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        # Load embedding model
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize variables
        self.df = None
        self.index = None
        self.embeddings = None

        # Load and process data
        self.load_data()
        self.create_embeddings()
        
    def load_data(self):
        """Load the medical FAQ data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} medical FAQs")
        
        # Clean the data
        self.df = self.df.dropna(subset=['Question', 'Answer'])
        
        # Combine question and answer for better context
        self.df['combined_text'] = self.df['Question'] + " " + self.df['Answer']
        print(f"After cleaning: {len(self.df)} FAQs ready")
        
    def create_embeddings(self):
        """Create embeddings for all questions and answers"""
        embeddings_path = 'embeddings/medical_embeddings.pkl'
        index_path = 'embeddings/faiss_index.index'
        
        # Create embeddings directory if it doesn't exist
        os.makedirs('embeddings', exist_ok=True)
        
        if os.path.exists(embeddings_path) and os.path.exists(index_path):
            print("Loading existing embeddings...")
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            self.index = faiss.read_index(index_path)
            print("Embeddings loaded successfully!")
        else:
            print("Creating embeddings... This might take a few minutes.")
            
            # Create embeddings for all combined texts
            texts = self.df['combined_text'].tolist()
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            # Save embeddings and index
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            faiss.write_index(self.index, index_path)
            print("Embeddings created and saved!")
            
        print("RAG system ready!")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str, float]]:
        """Retrieve most relevant documents for a query"""
        # Encode the query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in the index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant documents
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.df):  # Safety check
                row = self.df.iloc[idx]
                relevant_docs.append((
                    row['qtype'] if 'qtype' in self.df.columns else "General",
                    row['Question'], 
                    row['Answer'],
                    float(score)
                ))
        
        return relevant_docs
    
    def generate_answer(self, query: str, relevant_docs: List[Tuple[str, str, str, float]]) -> str:
        """Generate answer using Gemini with retrieved context"""

        # Prepare context from relevant documents
        context = ""
        for i, (qtype, question, answer, score) in enumerate(relevant_docs):
            context += f"\nContext {i+1} (Type: {qtype}):\nQ: {question}\nA: {answer}\n"

        # Create prompt
        prompt = f"""You are a helpful medical assistant. Answer the user's question based on the provided medical context. 
Be accurate, clear, and helpful. If the context doesn't fully answer the question, say so and provide what information you can.

Context from medical knowledge base:
{context}

User Question: {query}

Please provide a clear, helpful answer based on the context above:"""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text

            # Add disclaimer if not already present
            if answer and "consult" not in answer.lower() and "doctor" not in answer.lower():
                answer += "\n\n⚠️ Please consult with a healthcare professional for personalized medical advice."

            return answer

        except Exception as e:
            return f"Sorry, I couldn't generate an answer right now. Error: {str(e)}\n\nPlease check your GEMINI API key and try again."

    def chat(self, query: str) -> dict:
        """Main chat function that combines retrieval and generation"""
        if not query.strip():
            return {
                'query': query,
                'answer': "Please ask a medical question and I'll help you find relevant information.",
                'relevant_docs': [],
                'sources_used': 0
            }
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_docs(query)
            
            # Generate answer
            answer = self.generate_answer(query, relevant_docs)
            
            return {
                'query': query,
                'answer': answer,
                'relevant_docs': relevant_docs,
                'sources_used': len(relevant_docs)
            }
        
        except Exception as e:
            return {
                'query': query,
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'relevant_docs': [],
                'sources_used': 0
            }

# Test function for command line usage
def test_chatbot():
    """Simple test function to verify the chatbot works"""
    api_key = input("Enter your Gemini API key: ")
    csv_path = input("Enter path to your CSV file: ")
    
    chatbot = MedicalRAGChatbot(csv_path, api_key)
    
    print("\n" + "="*50)
    print("Medical Chatbot is ready! Type 'quit' to exit.")
    print("="*50)
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using the Medical Chatbot!")
            break
        
        response = chatbot.chat(query)
        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources used: {response['sources_used']}")

if __name__ == "__main__":
    test_chatbot()
