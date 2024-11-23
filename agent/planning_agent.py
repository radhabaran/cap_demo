from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory, SimpleMemory
import agent.router_agent as router_agent
import agent.product_review_agent as product_review_agent
import agent.generic_agent as generic_agent
import agent.composer_agent as composer_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None
agent = None

def initialize_planning_agent(llm_instance, chat_memory_instance, query_memory_instance):
    global llm, chat_memory, query_memory, agent
    
    llm = llm_instance
    chat_memory = chat_memory_instance
    query_memory = query_memory_instance
    
    # Initialize agents
    router_agent.initialize_router_agent(llm, chat_memory)
    product_review_agent.initialize_product_review_agent(llm, chat_memory)
    generic_agent.initialize_generic_agent(llm, chat_memory)
    # composer_agent.initialize_composer_agent(llm, memory)
    
    tools = [
        Tool(
            name="route_query",
            func=route_query,
            description="First step: Determine query type. Returns either 'product_review' or 'generic'"
        ),
        Tool(
            name="get_product_info",
            func=get_product_info,
            description="Use this for product-related queries about features, prices, availability, or reviews"
        ),
        Tool(
            name="handle_generic_query",
            func=handle_generic_query,
            description="Use this for general queries not related to products"
        ),
        Tool(
            name="compose_response",
            func=compose_response,
            description="Final step: Use this to format and enhance the response. After this step, return the response to main.py"
        )
    ]
    
    system_prompt = """You are a planning agent that processes user queries efficiently. Follow these steps:

    1. Follow these steps IN ORDER:
    - First use route_query
    - Then use get_product_info OR handle_generic_query based on route_query result. Return the response received first time from either get_product_info OR handle_generic_query and do not analyze the response any further.
    - ALWAYS end with compose_response
   
    2. MOST IMPORTANT: Your Final Answer MUST BE EXACTLY the Observation text returned by compose_response.
    - Do not summarize
    - Do not modify
    - Do not add your own conclusion
    - Simply copy the entire Observation from compose_response as your Final Answer

    For example:
    Thought: Need to route query
    Action: route_query
    Observation: product_review
    Thought: Getting product info
    Action: get_product_info
    Observation: [product details]
    Thought: Need to compose final response
    Action: compose_response
    Observation: [detailed response]
    Final Answer: [PASTE EXACT compose_response Observation here]
    """

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=chat_memory,
        system_message=system_prompt
    )
    logger.info("Planning agent initialized successfully")

def route_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return router_agent.classify_query(original_query)

def get_product_info(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    response = product_review_agent.process(original_query)
    print("********* response received by planning agent***********")
    print(response)
    # return product_review_agent.process(original_query)
    return response

def handle_generic_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return generic_agent.process(original_query)

def compose_response(response):
    return composer_agent.compose_response(response)

def execute(query):
    try:
        # Store original query
        query_memory.memories['original_query'] = query
        return agent.run(
            f"Process this user query: {query}"
        )
    except Exception as e:
        logger.error(f"Error in planning agent: {str(e)}")
        return f"Error in planning agent: {str(e)}"

def clear_context():
    if chat_memory:
        chat_memory.clear()
    if query_memory:
        query_memory.memories.clear()
    product_review_agent.clear_context()
    generic_agent.clear_context()