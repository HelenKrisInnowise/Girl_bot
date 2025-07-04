from contextlib import asynccontextmanager, contextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime, timedelta

# Import all necessary components from your modules
# Import initialize_mem0_clients function to call on startup
from modules.mem0_config import mem0_client, graph_mem0_client, initialize_mem0_clients 
from modules.llm_setup import (
    generate_dynamic_profile, get_user_personal_profile, get_user_personal_profile_graph,
     generate_proactive_query, generate_proactive_query_graph,
    detect_controversial_topic, get_system_prompt_template, llm, mood_llm
)
from modules.profiles import MAIN_CHARACTER_TRAITS, FORMALITY_LEVELS, COMMUNICATION_STYLES
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel as PydanticBaseModel # Alias to avoid conflict with FastAPI's BaseModel


@contextmanager
def lifespan(app: FastAPI):
    # Startup code (synchronous)
    print("FastAPI startup: Initializing Mem0 clients...")
    initialize_mem0_clients()  # Now synchronous
    print("FastAPI startup: Mem0 clients initialized.")
    yield
  
app = FastAPI(title="Girls Chatbot Backend")

# --- Pydantic Models for API Request/Response ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    prompt: str
    messages: List[Message]
    selected_traits: List[str]
    selected_formality: str
    selected_style: str
    dynamic_profile: Dict[str, str] # description, behavioral_traits

class ProfileGenerateRequest(BaseModel):
    traits: List[str]
    formality: str
    style: str

class ProactiveQueryRequest(BaseModel):
    user_id: str

class MoodHistoryRequest(BaseModel):
    user_id: str

class ChatResponse(PydanticBaseModel): # Changed from BaseModel to PydanticBaseModel
    response: str
    messages: List[Message]
    relevant_memories_str: str

# --- API Endpoints ---

@app.post("/generate_persona")
async def generate_persona_endpoint(request: ProfileGenerateRequest):
    try:
        # generate_dynamic_profile is synchronous, run in a thread pool
        new_profile = await asyncio.to_thread(generate_dynamic_profile, request.traits, request.formality, request.style)
        return {"dynamic_profile": new_profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating persona: {e}")

@app.post("/add_initial_memory")
async def add_initial_memory_endpoint(user_id: str, description: str):
    try:
        # Await the async mem0_client.add call
        await mem0_client.add(messages=[
            {"role": "assistant", "content": f"I am a chatbot designed to embody various girl personas. Use the sidebar to configure my personality! Initial persona: {description}"}
        ], user_id=user_id)
        return {"status": "success", "message": "Initial memory added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add initial memory to Mem0: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    prompt = request.prompt
    messages = request.messages
    dynamic_profile = request.dynamic_profile

    # Controversial Topic Detection
    controversial_analysis_result = await detect_controversial_topic(prompt) # Await async function
    if controversial_analysis_result.get('is_controversial'):
        category = controversial_analysis_result.get('category', 'undefined')
        # reason = controversial_analysis_result.get('reason', 'due to its content.')
        refusal_message = f"I cannot discuss topics related to **{category}**... Please let's move on to another topic."
        
        # Add refusal memory to Mem0 (Vector)
        try:
            await mem0_client.add(messages=[{"role": "assistant", "content": f"Refused to discuss controversial topic: {category} for user message: '{prompt}'"}],
                            user_id=user_id,
                            categories=["chatbot_interactions"])
        except Exception as add_refusal_e:
            print(f"Could not add refusal memory to Mem0: {add_refusal_e}")
        
        # Return refusal response
        messages.append(Message(role="assistant", content=refusal_message))
        # return ChatResponse(response=refusal_message, messages=messages, relevant_memories_str="N/A", user_mood_str="N/A", user_intent_str="N/A")
        return ChatResponse(response=refusal_message, messages=messages, relevant_memories_str="N/A")
    # Add user message to Mem0 (Cloud Vector Database)
    try:
        await mem0_client.add(messages=[{"role": "user", "content": prompt}], user_id=user_id)
    except Exception as e:
        print(f"Could not add user message to Mem0 (Cloud Vector Database): {e}")
    
    # Add user message to Mem0 (Graph Memory)
    try:
        # For graph memory, we add the raw message and Mem0's internal LLM extracts entities based on GRAPH_CUSTOM_PROMPT
        await graph_mem0_client.add(prompt, user_id=user_id)
    except Exception as e:
        print(f"Could not add user message to Mem0 (Graph Memory): {e}")


    # Retrieve relevant memories (from vector store)
    relevant_memories_str = "No relevant memories found."
    try:
        search_results = await mem0_client.search(query=prompt, user_id=user_id, limit=3) # Await async function
        if search_results:
            relevant_memories_str = "\n".join([f"- {m['memory']}" for m in search_results])
    except Exception as e:
        print(f"Could not perform Mem0 search: {e}")

    # Adaptive Response Generation
    profile_desc = dynamic_profile['description']
    profile_traits = dynamic_profile['behavioral_traits']

    system_prompt_template = get_system_prompt_template()

    system_message_content = system_prompt_template.format(
        persona_name="Dynamically Generated Persona",
        profile_description=profile_desc,
        profile_behavioral_traits=profile_traits,
        relevant_memories=relevant_memories_str,

    )

    llm_messages = [
        SystemMessage(content=system_message_content)
    ]
    # Reconstruct messages for LLM
    for msg in messages:
        if msg.role == "user":
            llm_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            llm_messages.append(AIMessage(content=msg.content))
    
    llm_messages.append(HumanMessage(content=prompt)) # Add current user prompt

    assistant_response = await asyncio.to_thread(llm.invoke, llm_messages) # Run synchronous LLM call in thread
    assistant_response_content = assistant_response.content

    messages.append(Message(role="assistant", content=assistant_response_content))

    return ChatResponse(
        response=assistant_response_content,
        messages=messages,
        relevant_memories_str=relevant_memories_str,
    )

@app.post("/get_user_profile_vector")
async def get_user_profile_vector_endpoint(user_id: str):
    try:
        profile_summary = await get_user_personal_profile(mem0_client, user_id) # Await
        return {"profile_summary": profile_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching/summarizing vector profile: {e}")

@app.post("/get_user_profile_graph")
async def get_user_profile_graph_endpoint(user_id: str):
    try:
        profile_summary_graph = await get_user_personal_profile_graph(graph_mem0_client, user_id) # Await
        return {"profile_summary_graph": profile_summary_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching/summarizing graph profile: {e}")


@app.post("/generate_proactive_query_vector")
async def generate_proactive_query_vector_endpoint(user_id: str):
    try:
        proactive_query = await generate_proactive_query(mem0_client, user_id) # Await
        return {"proactive_query": proactive_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proactive query (Vector): {e}")

@app.post("/generate_proactive_query_graph")
async def generate_proactive_query_graph_endpoint(user_id: str):
    try:
        proactive_query = await generate_proactive_query_graph(graph_mem0_client, user_id) # Await
        return {"proactive_query": proactive_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proactive query (Graph): {e}")

@app.post("/get_mood_history")
async def get_mood_history_endpoint(user_id: str):
    try:
        # Mood score map for re-use
        MOOD_SCORE_MAP = {
            "joyful": 5, "excited": 4, "neutral": 3, "confused": 2.5, "surprised": 3.5,
            "fearful": 1.5, "anxious": 1.7, "sad": 1, "angry": 0.5, "disgusted": 0
        }

        month_ago = datetime.now() - timedelta(days=30)
        month_ago_iso = month_ago.isoformat(timespec='seconds') + 'Z' 
        filters = {
            "AND": [
                {"user_id": user_id},
                {"created_at": {"gte": month_ago_iso}},
                {"categories": {"contains": "user_mood"}}
            ]
        }
        all_mood_memories = await mem0_client.get_all(version="v2", filters=filters, page_size=50) # Await

        mood_data_for_chart = []
        for memory in all_mood_memories:
            mood_text = memory.get('memory', '')
            detected_mood = "neutral"

            if mood_text:
                try:
                    parsed_mood_raw = await asyncio.to_thread(mood_llm.invoke, mood_text) # Run synchronous LLM call in thread
                    parsed_mood_data = {}
                    if isinstance(parsed_mood_raw, dict) and 'parsed' in parsed_mood_raw and isinstance(parsed_mood_raw['parsed'], PydanticBaseModel):
                        parsed_mood_data = parsed_mood_raw['parsed'].model_dump()
                    elif isinstance(parsed_mood_raw, PydanticBaseModel):
                        parsed_mood_data = parsed_mood_raw.model_dump()
                    
                    if parsed_mood_data and 'mood' in parsed_mood_data:
                        detected_mood = parsed_mood_data['mood'].lower()
                        
                except Exception as parse_e:
                    print(f"Could not parse mood from memory '{mood_text}': {parse_e}. Defaulting to neutral.")
                    detected_mood = "neutral"

            created_at_iso = memory.get('created_at')
            if created_at_iso:
                timestamp = datetime.fromisoformat(created_at_iso.split('.')[0]).replace(tzinfo=None)
            else:
                timestamp = datetime.now()
            
            mood_data_for_chart.append({
                "time": timestamp.isoformat(), # Convert datetime to string for JSON serialization
                "mood": detected_mood,
                "mood_score": MOOD_SCORE_MAP.get(detected_mood, 3)
            })
        
        return {"mood_history_data": mood_data_for_chart}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching mood history: {e}")

