import os
from dotenv import load_dotenv
from mem0 import AsyncMemory 
from mem0.configs.base import MemoryConfig
from mem0 import AsyncMemoryClient
import asyncio

load_dotenv()
os.environ["MEM0_API_KEY"] = os.getenv("MEM0_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Get all necessary environment variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MEM0_API_KEY = os.getenv("MEM0_API_KEY") # Ensure MEM0_API_KEY is loaded for mem0 cloud vector store
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME") # For explicit embedder config

# Basic validation for critical keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env. Please provide your OpenAI API key.")
if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is not set in .env. Please provide your Mem0 Cloud API key.")
if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("Neo4j environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) are not set. Graph Memory cannot be initialized.")
if not OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME:
    # Fallback for embedding model if not explicitly set, but warn
    print("Warning: OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME not set. Defaulting to 'text-embedding-3-small'.")
    OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME = "text-embedding-3-small"

# Define custom categories for Mem0
MEM0_CUSTOM_CATEGORIES = [
    {"personal_details": "Tracks user's name, age, gender, location, occupation, or any other direct personal identification."},
    {"intimate_life": "Details about user's romantic/sexual relationships, preferences, and experiences (only if explicitly mentioned)."},
    {"user_interests": "Tracks user's hobbies, passions, and preferred activities."},
    {"user_preferences": "General preferences or dislikes mentioned by the user (excluding intimate preferences)."},
    {"user_mood": "Tracks the user's emotional state including detected mood, its intensity, and reasons."},
    {"life_events": "Significant personal milestones like graduations, relocations, career changes (excluding intimate events)."},
    {"professional_details": "Information about user's current job, position, company, salary, work schedule."},
    {"daily_routine": "Information about user's regular schedule, habits, and daily activities."},
    {"achievements": "User's accomplishments in any area (professional, personal, sports, etc.)."},
    {"challenges": "Difficulties or problems the user is currently facing or has overcome."},
    {"future_plans": "User's goals, dreams, and plans for near or distant future."},
    {"relationships": "Information about user's family, friends, and social circle (excluding romantic/sexual partners)."},
    {"opinions": "User's views and beliefs on various topics (politics, ethics, philosophy, etc.)."},
    {"learning": "Topics/skills the user is currently learning or wants to learn."},
    {"favorites": "User's favorite items in different categories (books, movies, brands, etc.)."},
    {"pet_peeves": "Things that consistently annoy or bother the user."},
    {"bucket_list": "Things the user wants to experience or achieve in their lifetime."},
    {"health": "User's physical and mental health information, excluding sexual health."},
    {"sexual_health": "User's sexual wellbeing, preferences, and related health matters (if explicitly mentioned)."},
    {"chatbot_interactions": "Memories related to how the user interacts with the chatbot itself or its functionalities."},
]

MEM0_CUSTOM_INSTRUCTIONS = """
Your Task: Accurately extract and categorize information from user conversations with special attention to sensitive topics.

Enhanced Categorization Guidelines:

Personal Domain:
- **personal_details**: Name, age, gender, physical characteristics, location, contact details.
- **relationships**: Family members, friends, colleagues; nature of platonic relationships.
- **life_events**: Non-intimate milestones like graduations, relocations, career changes.
- **daily_routine**: Typical day structure excluding intimate activities.

Intimate Domain (handle with extra sensitivity):
- **intimate_life**: Current/past romantic/sexual relationships, dating preferences, orientation.
- **sexual_health**: Sexual wellbeing, preferences, contraceptive methods, related issues.

Professional Domain:
- **professional_details**: Current job, position, company, salary, work schedule.
- **achievements**: Promotions, awards, completed projects, publications.
- **challenges**: Work problems, difficult projects, conflicts at workplace.
- **future_plans**: Career goals, desired positions, retirement plans.

Leisure Domain:
- **user_interests**: Hobbies, sports, creative activities, club memberships.
- **travel**: Past trips, dream destinations, travel preferences.
- **entertainment**: Favorite shows, books, games, content preferences.

Personal Development:
- **learning**: Courses, skills being acquired, educational goals.
- **health**: General health, exercise, diets, medical conditions (excluding sexual health).
- **bucket_list**: Life goals and experiences user wants to have.

Preferences and Identity:
- **user_preferences**: Likes/dislikes (food, music, etc. excluding intimate preferences).
- **opinions**: Stances on social issues, political views, ethical beliefs.
- **favorites**: Preferred brands, colors, styles, artists, etc.
- **pet_peeves**: Specific annoyances and irritants.

Emotional Domain:
- **user_mood**: Current emotional state with context and intensity.
- **milestones**: Emotional significant events and their impact.

Chatbot_interactions
**chatbot_interactions**: Note down feedback or observations about the chatbot's behavior or functionality.

Special Handling Rules:
1. Context Sensitivity: Relationship status goes to:
   - 'relationships' if mentioned casually ("my friend X")
   - 'intimate_life' if romantic/sexual context ("my partner X")
2. Health Separation: General health vs sexual health must be strictly divided.
3. Preference Grading: Store intimacy-related preferences only in 'intimate_life' category.

Additional General Rules:
1. Temporal Tagging: Always note timeframe for sensitive information.
2. Confidence Scoring: For intimate categories, require higher certainty.
3. Relationship Mapping: Clearly distinguish platonic vs intimate relations.
4. Contradiction Resolution: For sensitive data, keep historical versions longer.

"""

# --- Mem0 Configuration for the Cloud Vector Store (mem0_client) ---
mem0_cloud_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4"
        }
    },
}

# --- Mem0 Configuration for the Graph Store (graph_mem0_client) ---
mem0_graph_config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URI", "bolt://neo4j:7687"),  # Fallback to default if not set
            "username": os.getenv("NEO4J_USER", "neo4j"),  # Default user is 'neo4j'
            "password": os.getenv("NEO4J_PASSWORD"),  # Required, no default
            "database": os.getenv("NEO4J_DATABASE", "neo4j"),  # Fallback to default DB
        }
    },
    "embedding": {
        "model": "text-embedding-3-small"
    },
    "version": "v1.1"
}
# Initialize MemoryConfig with your custom config
custom_config = MemoryConfig(config=mem0_graph_config)

mem0_client = None
graph_mem0_client = None


# def initialize_mem0_clients():
#     global mem0_client, graph_mem0_client # Declare global to modify them
#     try:
#         mem0_client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))
#         graph_mem0_client = AsyncMemory(config=custom_config)
#         mem0_client.update_project(
#             custom_categories=MEM0_CUSTOM_CATEGORIES,
#             custom_instructions=MEM0_CUSTOM_INSTRUCTIONS
#         )
#         print("Mem0 (Vector) project updated with custom categories and instructions.")

#     except Exception as e:
#         print(f"Failed to initialize Mem0 clients: {e}. Check OpenAI API Key, Mem0 API Key, Neo4j configs.")
#         # Re-raise the exception to indicate a critical startup failure
#         raise
#     return mem0_client, graph_mem0_client 

# mem0_client, graph_mem0_client = initialize_mem0_clients()
async def create_mem0_clients():
    """Factory function to create and initialize clients"""
    mem0_client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))
    graph_mem0_client = AsyncMemory(config=custom_config)
    
    await mem0_client.update_project(
        custom_categories=MEM0_CUSTOM_CATEGORIES,
        custom_instructions=MEM0_CUSTOM_INSTRUCTIONS
    )
    return mem0_client, graph_mem0_client