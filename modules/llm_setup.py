import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from modules.pydantic_models import UserProfile, ControversialTopicAttributes, MoodAttributes
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env.")


llm = ChatOpenAI(
    model="gpt-4",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

mood_llm = llm.with_structured_output(MoodAttributes, method="function_calling", include_raw=True)

class DynamicProfileOutput(BaseModel):
    description: str = Field(description="A concise description of the chatbot persona based on the selected traits, formality, and style.")
    behavioral_traits: str = Field(description="A detailed explanation of how the chatbot will behave, its tone, and interaction style, derived from the selected characteristics.")

profile_generator_llm = llm.with_structured_output(DynamicProfileOutput, method="function_calling", include_raw=False)
user_profile_llm = llm.with_structured_output(UserProfile, method="function_calling", include_raw=False)
controversial_llm = llm.with_structured_output(ControversialTopicAttributes, method="function_calling", include_raw=True)

def generate_dynamic_profile(traits: list[str], formality: str, style: str) -> dict:
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

async def get_user_personal_profile(mem0_client_instance, user_id: str) -> dict: # Made async, takes client
    if not user_id: return {"name": None, "interests": [], "preferences": [], "summary": "User ID is missing."}
    
    profile_filters = {
        "AND": [
            {"user_id": user_id},
            {"OR": [
                {"categories": {"contains": "personal_details"}},
                {"categories": {"contains": "user_interests"}},
                {"categories": {"contains": "user_preferences"}},
                {"categories": {"contains": "relationships"}},
                {"categories": {"contains": "opinions"}}
            ]}
        ]
    }
    personal_memories = await mem0_client_instance.get_all(version="v2", filters=profile_filters, page_size=30) # Await call

    if not personal_memories:
        return {"name": None, "interests": [], "preferences": [], "summary": "No personal information found."}
    
    memories_text = "\n".join([m['memory'] for m in personal_memories])

    prompt = f"""
    Based on the following fragmented user memories, synthesize a coherent User Personal Profile.
    Extract the user's name, a list of their main interests, a list of other general preferences, and a brief overall summary.
    

    User Memories:
    {memories_text}

    Output should strictly adhere to the UserProfile Pydantic model and it must naturally match the User Memories's language.
    """
    try:
        profile_summary = user_profile_llm.invoke(prompt)
        return profile_summary.model_dump()
    except Exception as e:
        print(f"Error generating user personal profile: {e}")
        return {"name": None, "summary": f"Could not generate profile: {e}"}

async def generate_proactive_query(mem0_client_instance, user_id: str) -> str: # Made async, takes client
    ten_days_ago = datetime.now() - timedelta(days=10)
    ten_days_ago_iso = ten_days_ago.isoformat(timespec='seconds') + 'Z' 
    filters = {
        "AND": [
            {"user_id": user_id},
            {"created_at": {"gte": ten_days_ago_iso}},
            {"OR": [
                {"categories": {"contains": "life_events"}},
                {"categories": {"contains": "daily_routine"}},
                {"categories": {"contains": "relationships"}}, 
                {"categories": {"contains": "professional_details"}},
            ]}
        ]
    }
    recent_relevant_memories = await mem0_client_instance.get_all(version="v2", filters=filters, page_size=15) # Await call

    if not recent_relevant_memories:
        return "It's been a while! How are you doing today? Anything new or interesting happening?"

    memories_content = "\n".join([f"- {m['memory']}" for m in recent_relevant_memories])

    prompt = f"""
    Based on the following recent memories about the user's life events and daily routines,
    formulate a message with a question or a phrase implying a response.
    The goal is to proactively and empathetically re-engage the user, as if you haven't chatted for a while.
    Make the message sound natural, friendly, and caring, it must naturally match the User Memories's language.

    Recent User Memories (from last 10 days):
    {memories_content}

    Example of desired output:
    "Hey! I was just thinking about you. How's that new project going that you mentioned?"
    "It's been a bit quiet! Hope you're doing well. Did you get a chance to try that new recipe you were excited about?"
    "Long time no chat! How are things with your morning runs these days? Still enjoying them?"
    """
    try:
        proactive_response = llm.invoke(prompt)
        return proactive_response.content
    except Exception as e:
        print(f"Error generating proactive query: {e}")
        return "I'm having a bit of trouble coming up with a new topic right now. Is there anything specific you'd like to talk about?"

async def detect_controversial_topic(text: str) -> dict: # Made async
    prompt = f"""
    Analyze the following user message to determine if it discusses a controversial topic.
    Controversial topics include: politics, religion, sexual content, violence, hate speech.

    Message: "{text}"

    Output should strictly adhere to the ControversialTopicAttributes Pydantic model.
    - Set 'is_controversial' to True if a controversial topic is detected.
    - Set 'category' to the primary controversial category (e.g., 'politics', 'religion', 'sexual', 'violence', 'hate_speech'). If multiple, pick the most dominant. If not controversial, set to 'none'.
    - Provide a short "refusal_message" if it is controversial.
    - "refusal_message" must naturally match the input message's language
    """
    try:
        controversial_analysis_raw = controversial_llm.invoke(prompt)
        if isinstance(controversial_analysis_raw, dict) and 'parsed' in controversial_analysis_raw and isinstance(controversial_analysis_raw['parsed'], BaseModel):
            return controversial_analysis_raw['parsed'].model_dump()
        elif isinstance(controversial_analysis_raw, BaseModel):
            return controversial_analysis_raw.model_dump()
        else:
            print(f"Warning: Controversial analysis result not a recognizable Pydantic model or dict with 'parsed'. Type: {type(controversial_analysis_raw)}")
            return {"is_controversial": False, "category": "none", "refusal_message": "Parsing issue or unknown format."}
    except Exception as e:
        print(f"Error during controversial topic detection: {e}")
        return {"is_controversial": False, "category": "none", "refusal_message": f"Detection failed: {e}"}

    
def get_system_prompt_template():
    return """
    You are a chatbot persona. Your identity and behavior are defined by the following profile:
    Core Identity: {profile_description}
    Behavioral Traits: {profile_behavioral_traits}

    **Relevant Memories from Mem0 (for additional context, integrate naturally):**
    {relevant_memories}

    **Instructions for your response:**
    - Always embody the persona defined by your core identity and behavioral traits.
    - Integrate relevant information from the "Relevant Memories" section if it helps make your response more contextual or personalized, but do not directly state that you got this information from memory.
    - Keep responses engaging and relevant to the conversation.
    - Maintain conversational flow.
    """
