import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, SimpleMemory
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os
import agent.planning_agent as planning_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None

def initialize_components():
    global llm, chat_memory, query_memory
    load_dotenv()
    api_key = os.environ['OA_API']           
    os.environ['OPENAI_API_KEY'] = api_key
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key
    )

    # Initialize memories
    chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    query_memory = SimpleMemory()

    # Initialize planning agent with both memories
    planning_agent.initialize_planning_agent(llm, chat_memory, query_memory)

    logger.info("Components initialized successfully")

def process_query(query, history):
    try:
        # Restore chat history from Gradio's history
        if history:
            for human_msg, ai_msg in history:
                if chat_memory and hasattr(chat_memory, 'chat_memory'):
                    chat_memory.chat_memory.add_user_message(human_msg)
                    chat_memory.chat_memory.add_ai_message(ai_msg)
        
        # Store original query in query memory
        query_memory.memories['original_query'] = query
        
        # Execute query through planning agent
        response = planning_agent.execute(query)
        
        # Add current interaction to chat memory
        if chat_memory and hasattr(chat_memory, 'chat_memory'):
            chat_memory.chat_memory.add_user_message(query)
            chat_memory.chat_memory.add_ai_message(response)
        
        return response

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"Error details: {str(e)}")

        if chat_memory and hasattr(chat_memory, 'chat_memory'):
            chat_memory.chat_memory.add_user_message(query)
            chat_memory.chat_memory.add_ai_message(error_msg)

        return error_msg

def clear_context():
    planning_agent.clear_context()
    chat_memory.clear()
    query_memory.memories.clear()
    return [], []

def create_gradio_app():
    from app import create_interface
    return create_interface(process_query, clear_context)

def main():
    """Main application entry point"""
    try:
        initialize_components()
        app = create_gradio_app()
        app.queue()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()