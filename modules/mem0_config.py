
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
    {"sexual_health": "User's sexual wellbeing, preferences, and related health matters (if explicitly mentioned)."}
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
# 6. Overwrite previous memories if new information contradicts old, for the same type of detail (e.g., if user changes their favorite color).


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



