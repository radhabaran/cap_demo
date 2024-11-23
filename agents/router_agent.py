from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, SimpleMemory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None
prompt = None

def initialize_router_agent(llm_instance, chat_memory_instance):
    global llm, chat_memory, prompt
    llm = llm_instance
    chat_memory = chat_memory_instance

    system_prompt = """You are an intelligent query classification system for an e-commerce platform.
    Your role is to accurately categorize incoming customer queries into one of two categories:

    1. product_review: 
       - Queries about product features, specifications, or capabilities
       - Questions about product prices and availability
       - Requests for product reviews or comparisons
       - Questions about product warranties or guarantees
       - Inquiries about product shipping or delivery
       - Questions about product compatibility or dimensions
       - Requests for recommendations between products

    2. generic:
       - General customer service inquiries
       - Account-related questions
       - Technical support issues not related to specific products
       - Website navigation help
       - Payment or billing queries
       - Return policy questions
       - Company information requests
       - Non-product related shipping questions
       - Any other queries not directly related to specific products

    INSTRUCTIONS:
    - Analyze the input query carefully
    - Respond ONLY with either "product_review" or "generic"
    - Do not include any other text in your response
    - If unsure, classify as "generic"

    EXAMPLES:

    User: "What are the features of the Samsung Galaxy S21?"
    Assistant: product_review

    User: "How much does the iPhone 13 Pro Max cost?"
    Assistant: product_review

    User: "Can you compare the Dell XPS 15 with the MacBook Pro?"
    Assistant: product_review

    User: "Is the Sony WH-1000XM4 headphone available in black?"
    Assistant: product_review

    User: "What's the battery life of the iPad Pro?"
    Assistant: product_review

    User: "I need help resetting my password"
    Assistant: generic

    User: "Where can I view my order history?"
    Assistant: generic

    User: "How do I update my shipping address?"
    Assistant: generic

    User: "What are your return policies?"
    Assistant: generic

    User: "I haven't received my refund yet"
    Assistant: generic

    User: "Do you ship internationally?"
    Assistant: generic

    User: "Can you recommend a good gaming laptop under $1000?"
    Assistant: product_review

    User: "What's the warranty period for electronics?"
    Assistant: generic

    User: "Is the Instant Pot dishwasher safe?"
    Assistant: product_review

    User: "How do I track my order?"
    Assistant: generic
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    logger.info("Router agent initialized successfully")


def classify_query(query):
    try:
        # Create chain with memory
        chain = prompt | llm

        # Add query to chat history before classification
        if chat_memory and hasattr(chat_memory, 'chat_memory'):
            chat_memory.chat_memory.add_user_message(query)

        # Classify the query
        response = chain.invoke({"input": query})
        category = response.content.strip().lower()
        
        # Validate category
        if category not in ["product_review", "generic"]:
            category = "generic"  # Default fallback


        # Add classification result to chat history
        if chat_memory and hasattr(chat_memory, 'chat_memory'):
            chat_memory.chat_memory.add_ai_message(f"Query classified as: {category}")
        
        logger.info(f"Query: {query}")
        logger.info(f"Classification: {category}")
        print("**** in router agent****")
        print("query :", query)
        print("category :", category)

        return category

    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return "generic"  # Default fallback on error


def get_classification_history():
    """Retrieve classification history from memory"""
    if chat_memory and hasattr(chat_memory, 'chat_memory'):
        return chat_memory.chat_memory.messages
    return []


def clear_context():
    """Clear all memory contexts"""
    if chat_memory:
        chat_memory.clear()
    logger.info("Router agent context cleared")