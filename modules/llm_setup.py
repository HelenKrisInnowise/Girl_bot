
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from modules.pydantic_models import MoodAttributes, IntentAttributes, UserProfile, ControversialTopicAttributes 
from pydantic import BaseModel, Field 
from typing import List
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([OPENAI_API_KEY]):
    raise ValueError("One or more Azure OpenAI environment variables are not set.")

llm = ChatOpenAI(
    model="gpt-4",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)
# LLMs for structured output (mood and intent detection)
mood_llm = llm.with_structured_output(MoodAttributes, method="function_calling", include_raw=True)
intent_llm = llm.with_structured_output(IntentAttributes,  method="function_calling", include_raw=True)

# Pydantic model for dynamic profile generation (MOVED HERE from pydantic_models.py)
class DynamicProfileOutput(BaseModel):
    description: str = Field(description="A concise description of the chatbot persona based on the selected traits, formality, and style.")
    behavioral_traits: str = Field(description="A detailed explanation of how the chatbot will behave, its tone, and interaction style, derived from the selected characteristics.")

# LLM for dynamic profile generation
profile_generator_llm = llm.with_structured_output(DynamicProfileOutput, method="function_calling", include_raw=False)
user_profile_llm = llm.with_structured_output(UserProfile, method="function_calling", include_raw=False)
controversial_llm = llm.with_structured_output(ControversialTopicAttributes, method="function_calling", include_raw=True)

def generate_dynamic_profile(traits: list[str], formality: str, style: str) -> dict:
    """
    Generates a chatbot persona description and behavioral traits based on selected characteristics.
    """
    prompt = f"""
    Create a unique chatbot persona description and its behavioral traits based on the following characteristics:

    Main Character Traits: {', '.join(traits) if traits else 'None specified'}
    Formality Level: {formality}
    Communication Style: {style}

    Ensure the description is concise and captures the essence. The behavioral traits should explain how the chatbot will interact, its tone, and specific mannerisms based on these combined characteristics. The chatbot's name will be dynamically generated later, so do not include a specific name.

    Example:
    Main Character Traits: friendly, empathetic
    Formality Level: informal
    Communication Style: supportive

    Output:
    {{
        "description": "A warm and understanding chatbot that listens attentively.",
        "behavioral_traits": "Responds with encouraging words and uses casual language. Always tries to uplift the user and offer empathetic advice. May use emojis occasionally."
    }}
    """
    try:
        profile_output = profile_generator_llm.invoke(prompt)
        return profile_output.model_dump()
    except Exception as e:
        print(f"Error generating dynamic profile: {e}")
        return {
            "description": "A friendly and helpful chatbot.",
            "behavioral_traits": "Responds in a straightforward and polite manner."
        }

def get_user_personal_profile(user_memories: List[dict]) -> dict:
    """
    Generates a summarized user personal profile (name, interests, preferences)
    based on a list of relevant memories from Mem0.
    """
    if not user_memories:
        return {"name": None, "interests": [], "preferences": [], "summary": "No personal information found."}
    prompt = f"""
    Based on the following fragmented user memories, synthesize a coherent User Personal Profile.
    Extract the user's name, a list of their main interests, a list of other general preferences, and a brief overall summary.

    User Memories:
    {user_memories}

    Output should strictly adhere to the UserProfile Pydantic model.
    """
    try:
        profile_summary = user_profile_llm.invoke(prompt)
        return profile_summary.model_dump()
    except Exception as e:
        print(f"Error generating user personal profile: {e}")
        return {"name": None, "interests": [], "preferences": [], "summary": f"Could not generate profile: {e}"}

def get_user_personal_profile_graph(user_memories: List[dict]) -> str:
    """
    Generates a summarized user personal profile
    based on the hole Graph stored in Mem0.
    """
    if not user_memories:
        return {"No personal information found in graph."}
    prompt = f"""
    Analyze this graph, built from messages from the user to the bot and tell about the user everything you can understand from this graph.
    Create a psychological portrait of the user.

    User Graph:
    {user_memories}

    Output should strictly adhere to the UserProfile Pydantic model.
    """
    try:
        profile_summary = user_profile_llm.invoke(prompt)
        return profile_summary.model_dump()
    except Exception as e:
        print(f"Error generating user personal profile: {e}")
        return {"name": None, "interests": [], "preferences": [], "summary": f"Could not generate profile: {e}"}
    
def generate_proactive_query_graph(user_memories: List[dict]) -> str:
    if not user_memories:
        return "It's been a while! How are you doing today? Anything new or interesting happening?"

    prompt = f"""
    Analyze this graph, built from messages from the user to the bot and based on this graph
    formulate a message with a question or a phrase implying a response.
    The goal is to proactively and empathetically re-engage the user, as if you haven't chatted for a while.
    Make the message sound natural, friendly, and caring.

    User's memory graph:
    {user_memories}

    Example of desired output:
    "Hi! Was thinking about your trip plans—did you end up booking that getaway you were dreaming about?"
    "Hey you! How's your week been? Did you ever get around to reorganizing your workspace like you planned?"
    "Hi! Just checking in—how's everything going with your new team? Settling in okay?"
    "Hey there! How's the painting project coming along? Still finding time for it?"
    """
    try:
        proactive_response = llm.invoke(prompt) # Use main LLM for a natural language suggestion
        return proactive_response.content
    except Exception as e:
        print(f"Error generating proactive query: {e}")
        return "It's been a while! How are you doing today? Anything new or interesting happening?"
    
    
    
# Renamed and re-purposed function
def generate_proactive_query(recent_memories: List[dict]) -> str:
    """
    Generates a proactive and empathetic query to re-engage the user,
    based on recent life events and daily routines from Mem0.
    """
    if not recent_memories:
        return "It's been a while! How are you doing today? Anything new or interesting happening?"

    # Extract memory content and format it for the prompt
    memories_content = "\n".join([f"- {m['memory']}" for m in recent_memories])

    prompt = f"""
    Based on the following recent memories about the user's life events and daily routines,
    formulate a message with a question or a phrase implying a response.
    The goal is to proactively and empathetically re-engage the user, as if you haven't chatted for a while.
    Make the message sound natural, friendly, and caring.

    Recent User Memories (from last 10 days):
    {memories_content}

    Example of desired output:
    "Hey! I was just thinking about you. How's that new project going that you mentioned?"
    "It's been a bit quiet! Hope you're doing well. Did you get a chance to try that new recipe you were excited about?"
    "Long time no chat! How are things with your morning runs these days? Still enjoying them?"
    """
    try:
        proactive_response = llm.invoke(prompt) # Use main LLM for a natural language suggestion
        return proactive_response.content
    except Exception as e:
        print(f"Error generating proactive query: {e}")
        return "It's been a while! How are you doing today? Anything new or interesting happening?"

def detect_controversial_topic(text: str) -> dict:
    """
    Detects if the given text discusses a controversial topic (politics, religion, sexual, violence, hate speech).
    Returns a dictionary conforming to ControversialTopicAttributes.
    """
    prompt = f"""
    Analyze the following user message to determine if it discusses a controversial topic.
    Controversial topics include: politics, religion, sexual content, violence, hate speech.

    Message: "{text}"

    Output should strictly adhere to the ControversialTopicAttributes Pydantic model.
    - Set 'is_controversial' to True if a controversial topic is detected.
    - Set 'category' to the primary controversial category (e.g., 'politics', 'religion', 'sexual', 'violence', 'hate_speech'). If multiple, pick the most dominant. If not controversial, set to 'none'.
    - Provide a brief 'reason' if it is controversial.
    """
    try:
        controversial_analysis_raw = controversial_llm.invoke(prompt)
        if isinstance(controversial_analysis_raw, dict) and 'parsed' in controversial_analysis_raw and isinstance(controversial_analysis_raw['parsed'], BaseModel):
            return controversial_analysis_raw['parsed'].model_dump()
        elif isinstance(controversial_analysis_raw, BaseModel):
            return controversial_analysis_raw.model_dump()
        else:
            print(f"Warning: Controversial analysis result not a recognizable Pydantic model or dict with 'parsed'. Type: {type(controversial_analysis_raw)}")
            return {"is_controversial": False, "category": "none", "reason": "Parsing issue or unknown format."}
    except Exception as e:
        print(f"Error during controversial topic detection: {e}")
        return {"is_controversial": False, "category": "none", "reason": f"Detection failed: {e}"}


# System prompt template for adaptive response generation
def get_system_prompt_template():
    return """
    You are a chatbot persona. Your identity and behavior are defined by the following profile:
    Core Identity: {profile_description}
    Behavioral Traits: {profile_behavioral_traits}

    **Relevant Memories from Mem0 (for additional context, integrate naturally):**
    {relevant_memories}

    **Instructions for your response:**
    - Always embody the persona defined by your core identity and behavioral traits.
    - Adapt your tone and response style based on the "Detected Mood" and "Detected Intent" of the user. For example:
        - If the user is 'sad', offer support according to your persona's traits.
        - If the user's intent is a 'question', answer it within your persona.
        - If the user is 'angry', respond calmly or match their intensity if that aligns with your persona.
    - Integrate relevant information from the "Relevant Memories" section if it helps make your response more contextual or personalized, but do not directly state that you got this information from memory.
    - Keep responses engaging and relevant to the conversation.
    - Maintain conversational flow.
    - Do not explicitly mention 'mood', 'intent' detection, or 'memories' to the user in your natural conversation.
    """