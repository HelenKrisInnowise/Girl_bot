
# # modules/llm_setup.py

# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# # Import MoodAttributes and IntentAttributes directly, no longer DynamicProfileOutput
# from modules.pydantic_models import MoodAttributes, IntentAttributes
# from pydantic import BaseModel, Field # Import BaseModel and Field for DynamicProfileOutput

# load_dotenv() # Load environment variables

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
# OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL") # e.g., gpt-4

# if not all([OPENAI_API_KEY, OPENAI_API_ENDPOINT, OPENAI_MODEL_DEPLOYMENT_NAME, OPENAI_MODEL]):
#     raise ValueError("One or more Azure OpenAI environment variables are not set.")


# # Main LLM for generating chatbot responses
# llm = AzureChatOpenAI(
#     azure_deployment=OPENAI_MODEL_DEPLOYMENT_NAME,
#     api_key=OPENAI_API_KEY,
#     azure_endpoint=OPENAI_API_ENDPOINT,
#     api_version="2024-02-15-preview",
#     temperature=0.7 # Adjust as needed for creativity vs. consistency
# )

# # LLMs for structured output (mood and intent detection)
# mood_llm = llm.with_structured_output(MoodAttributes, method="function_calling", include_raw=True)
# intent_llm = llm.with_structured_output(IntentAttributes, method="function_calling", include_raw=True)

# # New Pydantic model for dynamic profile generation - MOVED HERE
# class DynamicProfileOutput(BaseModel):
#     description: str = Field(description="A concise description of the chatbot persona based on the selected traits, formality, and style.")
#     behavioral_traits: str = Field(description="A detailed explanation of how the chatbot will behave, its tone, and interaction style, derived from the selected characteristics.")

# # New LLM for dynamic profile generation
# profile_generator_llm = llm.with_structured_output(DynamicProfileOutput, method="function_calling", include_raw=False) # No need for raw output here

# def generate_dynamic_profile(traits: list[str], formality: str, style: str) -> dict:
#     """
#     Generates a chatbot persona description and behavioral traits based on selected characteristics.
#     """
#     prompt = f"""
#     Create a unique chatbot persona description and its behavioral traits based on the following characteristics:

#     Main Character Traits: {', '.join(traits) if traits else 'None specified'}
#     Formality Level: {formality}
#     Communication Style: {style}

#     Ensure the description is concise and captures the essence. The behavioral traits should explain how the chatbot will interact, its tone, and specific mannerisms based on these combined characteristics. The chatbot's name will be dynamically generated later, so do not include a specific name.

#     Example:
#     Main Character Traits: friendly, empathetic
#     Formality Level: informal
#     Communication Style: supportive

#     Output:
#     {{
#         "description": "A warm and understanding chatbot that listens attentively.",
#         "behavioral_traits": "Responds with encouraging words and uses casual language. Always tries to uplift the user and offer empathetic advice. May use emojis occasionally."
#     }}
#     """
#     try:
#         # Invoke the LLM to generate the profile
#         profile_output = profile_generator_llm.invoke(prompt)
#         return profile_output.model_dump()
#     except Exception as e:
#         print(f"Error generating dynamic profile: {e}")
#         return {
#             "description": "A friendly and helpful chatbot.",
#             "behavioral_traits": "Responds in a straightforward and polite manner."
#         }


# # System prompt template for adaptive response generation
# def get_system_prompt_template():
#     return """
#     You are a chatbot persona. Your identity and behavior are defined by the following profile:
#     Core Identity: {profile_description}
#     Behavioral Traits: {profile_behavioral_traits}

#     **User's Current Context:**
#     Detected Mood: {user_mood}
#     Detected Intent: {user_intent}

#     **Relevant Memories from Mem0 (for additional context, integrate naturally):**
#     {relevant_memories}

#     **Instructions for your response:**
#     - Always embody the persona defined by your core identity and behavioral traits.
#     - Adapt your tone and response style based on the "Detected Mood" and "Detected Intent" of the user. For example:
#         - If the user is 'sad', offer support according to your persona's traits.
#         - If the user's intent is a 'question', answer it within your persona.
#         - If the user is 'angry', respond calmly or match their intensity if that aligns with your persona.
#     - Integrate relevant information from the "Relevant Memories" section if it helps make your response more contextual or personalized, but do not directly state that you got this information from memory.
#     - Keep responses engaging and relevant to the conversation.
#     - Maintain conversational flow.
#     - Do not explicitly mention 'mood', 'intent' detection, or 'memories' to the user in your natural conversation.
#     """

# modules/llm_setup.py

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
# Import all Pydantic models from pydantic_models
from modules.pydantic_models import MoodAttributes, IntentAttributes, UserProfile
from pydantic import BaseModel, Field # Import BaseModel and Field for DynamicProfileOutput
from typing import List

load_dotenv() # Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") # e.g., gpt-4

if not all([OPENAI_API_KEY, OPENAI_API_ENDPOINT, OPENAI_MODEL_DEPLOYMENT_NAME, OPENAI_MODEL]):
    raise ValueError("One or more Azure OpenAI environment variables are not set.")


# Main LLM for generating chatbot responses
llm = AzureChatOpenAI(
    azure_deployment=OPENAI_MODEL_DEPLOYMENT_NAME,
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_ENDPOINT,
    api_version="2024-02-15-preview",
    temperature=0.7 # Adjust as needed for creativity vs. consistency
)

# LLMs for structured output (mood and intent detection)
mood_llm = llm.with_structured_output(MoodAttributes, method="function_calling", include_raw=True)
intent_llm = llm.with_structured_output(IntentAttributes, method="function_calling", include_raw=True)

# Pydantic model for dynamic profile generation (MOVED HERE from pydantic_models.py)
class DynamicProfileOutput(BaseModel):
    description: str = Field(description="A concise description of the chatbot persona based on the selected traits, formality, and style.")
    behavioral_traits: str = Field(description="A detailed explanation of how the chatbot will behave, its tone, and interaction style, derived from the selected characteristics.")

# LLM for dynamic profile generation
profile_generator_llm = llm.with_structured_output(DynamicProfileOutput, method="function_calling", include_raw=False)

# LLM for generating user profile summary from Mem0 data
user_profile_llm = llm.with_structured_output(UserProfile, method="function_calling", include_raw=False)

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

    memories_text = "\n".join([m['memory'] for m in user_memories])
    prompt = f"""
    Based on the following fragmented user memories, synthesize a coherent User Personal Profile.
    Extract the user's name, a list of their main interests, a list of other general preferences, and a brief overall summary.

    User Memories:
    {memories_text}

    Output should strictly adhere to the UserProfile Pydantic model.
    """
    try:
        profile_summary = user_profile_llm.invoke(prompt)
        return profile_summary.model_dump()
    except Exception as e:
        print(f"Error generating user personal profile: {e}")
        return {"name": None, "interests": [], "preferences": [], "summary": f"Could not generate profile: {e}"}

def suggest_conversation_topic(topic_memories: List[dict]) -> str:
    """
    Suggests a conversation topic based on a list of preferred topics from Mem0.
    """
    if not topic_memories:
        return "It seems we haven't discussed your favorite topics yet! What's on your mind today?"

    topics_list = [m['memory'] for m in topic_memories]
    prompt = f"""
    Based on the following list of topics the user has expressed interest in, suggest one interesting conversation topic.
    Make the suggestion sound natural and engaging, fitting a friendly chatbot persona.
    Do not just list the topics. Suggest one specific, engaging topic.

    User's Preferred Topics:
    {', '.join(topics_list)}

    Example: "Since you mentioned your love for space exploration, how about we dive into the latest Mars rover discoveries?"
    """
    try:
        suggestion_response = llm.invoke(prompt) # Use main LLM for a natural language suggestion
        return suggestion_response.content
    except Exception as e:
        print(f"Error suggesting topic: {e}")
        return "I'm having a bit of trouble coming up with a new topic right now. Is there anything specific you'd like to talk about?"

# System prompt template for adaptive response generation
def get_system_prompt_template():
    return """
    You are a chatbot persona. Your identity and behavior are defined by the following profile:
    Core Identity: {profile_description}
    Behavioral Traits: {profile_behavioral_traits}

    **User's Current Context:**
    Detected Mood: {user_mood}
    Detected Intent: {user_intent}

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

