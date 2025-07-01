
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from modules.pydantic_models import MoodAttributes, IntentAttributes, UserProfile, ControversialTopicAttributes 
from pydantic import BaseModel, Field 
from langchain.output_parsers.openai_tools import PydanticToolsParser
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
    print(user_memories)
    # memories_text = "\n".join([m['memory'] for m in user_memories])
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

def generate_proactive_query_graph2(mem0_client_instance, user_id: str, llm = llm) -> str:
    """
    Generates a personalized question based on the user's graph data analysis.

    Args:
        mem0_client_instance: Mem0 client instance
        user_id: User identifier
        llm: Initialized OpenAI client

    Returns:
        A string containing the personalized question
    """
    def analyze_graph_data(graph_data: Dict) -> Dict[str, List[str]]:
      
        user_prefix = f"user_id:_{user_id}"
        analysis = {
            'people': set(),       
            'activities': set(),    
            'relationships': [],     
            'timed_patterns': [],   
            'emotional_connections': []  
        }

        for rel in graph_data.get('relations', []):
            source = rel['source'].replace(user_prefix, 'You') if user_prefix in rel['source'] else rel['source']
            target = rel['target']
            rel_type = rel['relationship'].lower()

            if any(kwd in rel_type for kwd in ['of', 'with', 'has', 'is', 'knows', 'friend', 'mother', 'father']):
                for entity in [source, target]:
                    if entity.lower() not in ['week', 'month', 'day']:
                        analysis['people'].add(entity)


            if any(kwd in rel_type for kwd in ['spends', 'does', 'at', 'on', 'engaged', 'activity']):
                if target.lower() not in ['week', 'month', 'day']:
                    analysis['activities'].add(target)

            
            if source == 'You':
                analysis['relationships'].append({
                    'type': rel['relationship'],
                    'with': target,
                    'full': f"You {rel['relationship']} {target}"
                })

            
            if any(emo_word in rel_type for emo_word in ['love', 'like', 'hate', 'feel']):
                analysis['emotional_connections'].append(f"{source} {rel['relationship']} {target}")

            
            if any(time_word in target.lower() for time_word in ['week', 'month', 'day', 'year']):
                analysis['timed_patterns'].append(f"{source} {rel['relationship']} {target}")

        
        return {
            'people': list(analysis['people']),
            'activities': list(analysis['activities']),
            'relationships': analysis['relationships'],
            'emotional_connections': analysis['emotional_connections'],
            'timed_patterns': analysis['timed_patterns']
        }

    try:
        
        graph_data = mem0_client_instance.get_all(user_id=user_id)
        if not graph_data.get('results') and not graph_data.get('relations'):
            return "It's been a while! How have you been?"

        
        analysis = analyze_graph_data(graph_data)
        
        context_parts = []
        
        if analysis['people']:
            context_parts.append(f"Close contacts: {', '.join(analysis['people'])[:150]}")
        
        if analysis['relationships']:
            rels = [r['full'] for r in analysis['relationships'][-3:]]  # Последние 3 отношения
            context_parts.append(f"Your relationships: {'; '.join(rels)}")
        
        if analysis['emotional_connections']:
            context_parts.append(f"Emotional notes: {analysis['emotional_connections'][-1]}")
        
        if analysis['timed_patterns']:
            context_parts.append(f"Recurring: {analysis['timed_patterns'][-1]}")

        context = "\n".join(context_parts) if context_parts else "No recent updates"

        prompt = f"""Generate exactly ONE follow-up question based on:
        
        {context}
        
        Rules:
        1. Sound genuinely interested
        2. Reference specific relationships/activities
        3. Use 15-20 words max
        4. Focus on recent or emotional connections
        
        Good Examples:
        "How's your relationship with {analysis['people'][0]} been lately?"
        "Still doing {analysis['activities'][0]} regularly?"
        "How do you feel about {analysis['emotional_connections'][0].split()[-1]} now?"
        """
        
        response = llm.invoke(prompt)
        question = response.content.strip()
        
        # Проверка качества вопроса
        if not question or len(question.split()) > 25:
            return "How have you been lately? Anything new?"
        
        return question

    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        return "Long time no see! How are things going?"

def generate_proactive_query_graph(mem0_client_instance, user_id: str, llm = llm) -> str:
    """
    Generates a personalized question based on the user's graph data analysis.

    Args:
        mem0_client_instance: Mem0 client instance
        user_id: User identifier
        llm: Initialized DeepAI client

    Returns:
        A string containing the personalized question
    """
    def analyze_graph_data(graph_data: Dict) -> Dict[str, List[str]]:
        """Анализирует графовые данные и возвращает структурированную информацию"""
        user_prefix = f"user_id:_{user_id}"
        analysis = {
            'people': set(),
            'activities': set(),
            'timed_patterns': []
        }

        # Обработка отношений
        for rel in graph_data.get('relations', []):
            source = rel['source'].replace(user_prefix, 'You') if user_prefix in rel['source'] else rel['source']
            target = rel['target']
            rel_type = rel['relationship'].lower()

            # Собираем людей (исключая временные понятия)
            if any(kwd in rel_type for kwd in ['of', 'with', 'has', 'is', 'knows']):
                for person in [source, target]:
                    if person.lower() not in ['week', 'month', 'day', 'none']:
                        analysis['people'].add(person)

            # Собираем активности
            if any(kwd in rel_type for kwd in ['spends', 'does', 'at', 'on', 'engaged']):
                if target.lower() not in ['week', 'month', 'day']:
                    analysis['activities'].add(target)

            # Временные паттерны
            if any(time_word in target.lower() for time_word in ['week', 'month', 'day', 'year']):
                analysis['timed_patterns'].append(f"{source} {rel['relationship']} {target}")

        # Конвертируем множества в списки
        return {
            'people': list(analysis['people']),
            'activities': list(analysis['activities']),
            'timed_patterns': analysis['timed_patterns']
        }

    try:
        # Получаем данные из графа
        graph_data = mem0_client_instance.get_all(user_id = user_id)
        print(graph_data)
        if not graph_data.get('results') and not graph_data.get('relations'):
            return "It's been a while! How have you been?"

        # Анализируем граф
        analysis = analyze_graph_data(graph_data)
        
        # Формируем контекст
        context_parts = []
        if analysis['people']:
            context_parts.append(f"People: {', '.join(analysis['people'])[:200]}")
        if analysis['activities']:
            context_parts.append(f"Activities: {', '.join(analysis['activities'])[:200]}")
        if analysis['timed_patterns']:
            context_parts.append(f"Recent: {analysis['timed_patterns'][-1]}")
        
        context = "\n".join(context_parts) if context_parts else "No recent activity"

        print(context)
        # Генерируем вопрос
        prompt = f"""Generate exactly ONE natural follow-up question based on:
        
        {context}
        
        Guidelines:
        - Sound like a caring friend
        - Reference specific people/activities if possible
        - Keep it under 20 words
        - Use present tense
        
        Examples (DO NOT COPY):
        "How's your mother doing?"
        "Still meeting Julia for coffee weekly?"
        """
        
        response = llm.invoke(prompt)
        question = response.content.strip()
        
        # Проверка качества вопроса
        if not question or len(question.split()) > 25:
            return "How have you been lately?"
        
        return question

    except Exception as e:
        print(f"[ERROR] Question generation failed: {str(e)}")
        return "Long time no see! How are things going?"
    
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

