
import os
from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv() 

# Set environment variables required by Mem0 for Azure OpenAI if not already set
# Mem0's internal LLM will use these for its reasoning.
os.environ["LLM_AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LLM_AZURE_DEPLOYMENT"] = os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME")
os.environ["LLM_AZURE_ENDPOINT"] = os.getenv("OPENAI_API_ENDPOINT")
os.environ["LLM_AZURE_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

# Define custom categories for Mem0
MEM0_CUSTOM_CATEGORIES = [
    {"user_interests": "Tracks user's hobbies, passions, and preferred activities."},
    {"preferred_conversation_topics": "Records subjects or themes the user explicitly expresses interest in discussing."},
    {"chatbot_interactions": "Memories related to how the user interacts with the chatbot itself or its functionalities."},
    {"user_preferences": "General preferences or dislikes mentioned by the user."},
    {"user_mood": "Tracks the user's emotional state, including detected mood and its intensity, and reasons for their mood."}
]

# Define custom instructions for Mem0's internal processing
MEM0_CUSTOM_INSTRUCTIONS = """
Your Task: Accurately extract and categorize information from user conversations.

Guidelines for Categorization:
- **personal_details**: Extract and store the user's name, age, gender, location, occupation, or any other direct personal identification.
- **user_interests**: Identify and categorize any hobbies, passions, or activities the user mentions enjoying or being interested in (e.g., "I love playing guitar," "I'm into astronomy").
- **preferred_conversation_topics**: Extract and categorize any topics or subjects the user explicitly indicates they enjoy discussing or want to talk about more (e.g., "Let's talk about AI," "I'm interested in philosophy").
- **user_preferences**: Capture any general likes, dislikes, or opinions the user expresses (e.g., "I prefer coffee over tea," "I don't like horror movies").
- **chatbot_interactions**: Note down feedback or observations about the chatbot's behavior or functionality.
- **user_mood**: Identify the user's emotional state (e.g., joyful, sad, angry, neutral) and its intensity, as well as the likely reasons for that mood, if expressed. Store a concise summary of the mood detected in the user's message.

Important:
- Be precise and concise in the 'memory' field.
- Do not make assumptions. Only categorize information explicitly stated or strongly implied.
- Overwrite previous memories if new information contradicts old, for the same type of detail (e.g., if user changes their favorite color).
"""


# Configuration for Mem0's internal LLM to use Azure OpenAI
mem0_config = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME"), # Use deployment name here
            "temperature": 0.1, # Lower temperature for more consistent memory operations
            "max_tokens": 2000,
            "azure_kwargs": {
                "azure_deployment": os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME"),
                "api_version": os.getenv("OPENAI_API_VERSION"),
                "azure_endpoint": os.getenv("OPENAI_API_ENDPOINT"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        }
    }
}

# Initialize Mem0 client (without custom categories/instructions in constructor)
try:
    mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
    
    # Set custom categories and instructions at the project level AFTER initialization
    try:
        mem0_client.update_project(
            custom_categories=MEM0_CUSTOM_CATEGORIES,
            custom_instructions=MEM0_CUSTOM_INSTRUCTIONS
        )
        print("Mem0 project updated with custom categories and instructions.")
    except Exception as update_e:
        # Catch errors during update_project, but allow client init to succeed if possible
        print(f"Warning: Could not update Mem0 project settings: {update_e}")
        print("Ensure your API key has project write permissions.")
        # Re-raise only if client itself failed to init, not just the update
        pass # Do not re-raise, allow the app to continue without custom settings if update fails


except Exception as e:
    raise RuntimeError(f"Failed to initialize Mem0 client: {e}. Check MEM0_API_KEY and Azure OpenAI LLM configs.")



