# ***********************************************************************************************
# Instruction for using the program
# ***********************************************************************************************
# Please make sure the embeddings.npy file is available in data folder
# Please make sure the documents.pkl file is available in data folder
# Please set the path appropriately inside the program. You will find the below two statements 
# where you need to mention the correct path name.
# embedding_path = '/workspaces/IISC_cap_langchain/data/embeddings.npy'
# documents_path = '/workspaces/IISC_cap_langchain/documents.pkl'
# ***********************************************************************************************

import openai
import numpy as np
import pandas as pd
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import faiss
import warnings
import os

warnings.filterwarnings("ignore")
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
# vectorstore = None

def initialize_product_review_agent(llm_instance, memory_instance):
    """Initialize the product review agent with LLM and memory instances"""
    global llm, chat_memory

    llm = llm_instance
    chat_memory = memory_instance
 


def process(query):

    # Initialize the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    System_Prompt = """
    Role and Capabilities:
You are an AI customer service specialist for Amazon, focusing on the various products available in Amazon. Your primary functions are:
1. Providing accurate product information and pricing
2. Handling delivery-related queries
3. Addressing product availability
4. Offering technical support for electronics

Core Instructions:
1. Product Information:
   - Provide detailed specifications and features
   - Compare similar products when relevant
   - Only discuss products found in the provided context
   - Highlight key benefits and limitations

2. Price & Availability:
   - Quote exact prices from the provided context
   - Explain any pricing variations or discounts
   - Provide clear stock availability information
   - Mention delivery timeframes when available

3. Query Handling:
   - Address the main query first, then provide additional relevant information
   - For multi-part questions, structure answers in bullet points
   - If information is missing from context, explicitly state this
   - Suggest alternatives when a product is unavailable

Communication Guidelines:
1. Response Structure:
   - Start with a direct answer to the query
   - Provide supporting details and context
   - End with a clear next step or call to action
   - Include standard closing: "Thank you for choosing Amazon. Is there anything else I can help you with?"

2. Tone and Style:
   - Professional yet friendly
   - Clear and jargon-free language
   - Empathetic and patient
   - Concise but comprehensive

Limitations and Restrictions:
1. Only provide information present in the given context
2. Clearly state when information is not available
3. Never share personal or sensitive information
4. Don't make promises about delivery times unless explicitly stated in context

Error Handling:
1. Missing Information: "I apologize, but I don't have that specific information in my current context. Would you like me to provide related details about [alternative topic]?"
2. Out of Scope: "While I can't assist with [topic], I'd be happy to help you with electronics or home care products."
3. Technical Issues: "I apologize for any inconvenience. Could you please rephrase your question or provide more details?"

Response Format:
1. For product queries:
   - Product name and model
   - Price and availability
   - Key features
   - Recommendations if relevant

2. For service queries:
   - Current status
   - Next steps
   - Timeline (if available)
   - Contact options

Remember: Always verify information against the provided context before responding. Don't make assumptions or provide speculative information.

    """

    # Get existing chat history from memory
    chat_history = ""
    if chat_memory:
        messages = chat_memory.chat_memory.messages
        if messages:
            chat_history = "\nPrevious conversation:\n"
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    chat_history += f"Human: {messages[i].content}\n"
                    chat_history += f"Assistant: {messages[i+1].content}\n"

    # Check if embeddings already exist
    embedding_path = './data/embeddings.npy'
    documents_path = './documents.pkl'

    # Modify the get_embedding function to use LangChain's OpenAIEmbeddings
    def get_embedding(text, engine="text-embedding-ada-002"):
        return embeddings.embed_query(text)

    try:
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found at: {documents_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    if os.path.exists(embedding_path) and os.path.exists(documents_path):
        # Load existing embeddings and documents
        embeddings_list = np.load(embedding_path)
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)


    # Create FAISS index with faster search
    embeddings_np = np.array(embeddings_list).astype('float32')
    index=faiss.IndexFlatL2(len(embeddings_list[0]))
    index.add(embeddings_np)

    query_embedding = get_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)
    print("context retrieved :", context)
    print('*' * 100)

    # Include chat history in the prompt for context
    structured_prompt = f"""
    Context:
    {context}

    {chat_history}
    
    Current Query:
    {query}
    """

    print("structured prompt created :", structured_prompt)
    print('*' * 100)
    # Create messages for the chat model
    messages = [
        {"role": "system", "content": System_Prompt},
        {"role": "user", "content": structured_prompt}
    ]

    # For chat completion, you can use LangChain's ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
    response = chat_model.invoke(messages).content

    # Update memory
    if chat_memory:
        chat_memory.chat_memory.add_user_message(query)
        chat_memory.chat_memory.add_ai_message(response)

    logger.info(f"Successfully processed query: {query}")
    print("response returned by product_review_agent", response)
    return response


def clear_context():
    """Clear the conversation memory"""
    if chat_memory:
        chat_memory.clear()
        logger.info("Conversation context cleared")