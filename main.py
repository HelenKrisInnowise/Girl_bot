from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Annotated 
import asyncio
import uuid
from datetime import datetime, timedelta
from fastapi import Depends  # For dependency injection
from mem0 import AsyncMemoryClient, AsyncMemory  # Import your Mem0 client types

# Import client factory instead of direct clients
from modules.mem0_config import create_mem0_clients
from modules.llm_setup import (
    generate_dynamic_profile, get_user_personal_profile, get_user_personal_profile_graph,
    generate_proactive_query, generate_proactive_query_graph,
    detect_controversial_topic, get_system_prompt_template, llm, mood_llm
)
from modules.profiles import MAIN_CHARACTER_TRAITS, FORMALITY_LEVELS, COMMUNICATION_STYLES
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel as PydanticBaseModel

# --- Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and store clients
    print("FastAPI startup: Initializing Mem0 clients...")
    app.state.mem0_client, app.state.graph_mem0_client = await create_mem0_clients()
    print("FastAPI startup: Mem0 clients initialized.")
    yield
    # Cleanup code could go here if needed

app = FastAPI(title="Girls Chatbot Backend", lifespan=lifespan)

async def get_mem0_client(request: Request) -> AsyncMemoryClient:
    return request.app.state.mem0_client

async def get_graph_client(request: Request) -> AsyncMemory:
    return request.app.state.graph_mem0_client

# --- Pydantic Models ---
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
    dynamic_profile: Dict[str, str]

class ProfileGenerateRequest(BaseModel):
    traits: List[str]
    formality: str
    style: str

class ProactiveQueryRequest(BaseModel):
    user_id: str

class MoodHistoryRequest(BaseModel):
    user_id: str

class ChatResponse(PydanticBaseModel):
    response: str
    messages: List[Message]
    relevant_memories_str: str

# --- API Endpoints ---
@app.post("/generate_persona")
async def generate_persona_endpoint(request: ProfileGenerateRequest):
    try:
        new_profile = await asyncio.to_thread(
            generate_dynamic_profile, 
            request.traits, 
            request.formality, 
            request.style
        )
        return {"dynamic_profile": new_profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating persona: {e}")

@app.post("/add_initial_memory")
async def add_initial_memory_endpoint(
    user_id: str, 
    description: str,
    mem0_client: AsyncMemoryClient = Depends(get_mem0_client)
):
    try:
        await mem0_client.add(messages=[{
            "role": "assistant", 
            "content": f"I am a chatbot designed to embody various girl personas. Initial persona: {description}"
        }], user_id=user_id)
        return {"status": "success", "message": "Initial memory added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add initial memory: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    mem0_client: AsyncMemoryClient = Depends(get_mem0_client),
    graph_mem0_client: AsyncMemory = Depends(get_graph_client)
):
    try:
        # Controversial topic detection
        controversial_analysis_result = await detect_controversial_topic(request.prompt)
        if controversial_analysis_result.get('is_controversial'):
            category = controversial_analysis_result.get('category', 'undefined')
            refusal_message = f"I cannot discuss topics related to {category}..."
            
            await mem0_client.add(
                messages=[{
                    "role": "assistant", 
                    "content": f"Refused to discuss: {category} for message: '{request.prompt}'"
                }],
                user_id=request.user_id,
                categories=["chatbot_interactions"]
            )
            
            request.messages.append(Message(role="assistant", content=refusal_message))
            return ChatResponse(
                response=refusal_message,
                messages=request.messages,
                relevant_memories_str="N/A"
            )

        # Store messages
        await mem0_client.add(
            messages=[{"role": "user", "content": request.prompt}],
            user_id=request.user_id
        )
        await graph_mem0_client.add(request.prompt, user_id=request.user_id)

        # Get relevant memories
        relevant_memories_str = "No relevant memories found."
        try:
            search_results = await mem0_client.search(
                query=request.prompt,
                user_id=request.user_id,
                limit=3
            )
            if search_results:
                relevant_memories_str = "\n".join([f"- {m['memory']}" for m in search_results])
        except Exception as e:
            print(f"Search error: {e}")

        # Generate response
        system_message_content = get_system_prompt_template().format(
            persona_name="Dynamically Generated Persona",
            profile_description=request.dynamic_profile['description'],
            profile_behavioral_traits=request.dynamic_profile['behavioral_traits'],
            relevant_memories=relevant_memories_str,
        )

        llm_messages = [SystemMessage(content=system_message_content)]
        for msg in request.messages:
            llm_messages.append(
                HumanMessage(content=msg.content) if msg.role == "user" 
                else AIMessage(content=msg.content)
            )
        llm_messages.append(HumanMessage(content=request.prompt))

        assistant_response = await asyncio.to_thread(llm.invoke, llm_messages)
        response_content = assistant_response.content
        
        updated_messages = request.messages.copy()
        updated_messages.append(Message(role="assistant", content=response_content))

        return ChatResponse(
            response=response_content,
            messages=updated_messages,
            relevant_memories_str=relevant_memories_str,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get_user_profile_vector")
async def get_user_profile_vector_endpoint(
    user_id: str,
    mem0_client: Annotated[AsyncMemoryClient, Depends(get_mem0_client)]
):
    try:
        profile_summary = await get_user_personal_profile(mem0_client, user_id)
        return {"profile_summary": profile_summary}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching/summarizing vector profile: {str(e)}"
        )

@app.post("/get_user_profile_graph")
async def get_user_profile_graph_endpoint(
    user_id: str,
    graph_mem0_client: AsyncMemory = Depends(get_graph_client)
):
    try:
        profile_summary = await get_user_personal_profile_graph(graph_mem0_client, user_id)
        return {"profile_summary_graph": profile_summary}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching/summarizing graph profile: {str(e)}"
        )

@app.post("/generate_proactive_query_vector")
async def generate_proactive_query_vector_endpoint(
    user_id: str,
    mem0_client: AsyncMemoryClient = Depends(get_mem0_client)
):
    try:
        proactive_query = await generate_proactive_query(mem0_client, user_id)
        return {"proactive_query": proactive_query}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating proactive query (Vector): {str(e)}"
        )

@app.post("/generate_proactive_query_graph")
async def generate_proactive_query_graph_endpoint(
    user_id: str,
    graph_mem0_client: AsyncMemory = Depends(get_graph_client)
):
    try:
        proactive_query = await generate_proactive_query_graph(graph_mem0_client, user_id)
        return {"proactive_query": proactive_query}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating proactive query (Graph): {str(e)}"
        )

@app.post("/get_mood_history")
async def get_mood_history_endpoint(
    user_id: str,
    mem0_client: AsyncMemoryClient = Depends(get_mem0_client)
):
    MOOD_SCORE_MAP = {
        "joyful": 5, "excited": 4, "neutral": 3, "confused": 2.5, "surprised": 3.5,
        "fearful": 1.5, "anxious": 1.7, "sad": 1, "angry": 0.5, "disgusted": 0
    }

    try:
        # Calculate date range
        month_ago = datetime.now() - timedelta(days=30)
        month_ago_iso = month_ago.isoformat(timespec='seconds') + 'Z'
        
        # Build filters
        filters = {
            "AND": [
                {"user_id": user_id},
                {"created_at": {"gte": month_ago_iso}},
                {"categories": {"contains": "user_mood"}}
            ]
        }

        # Fetch mood memories
        memories = await mem0_client.get_all(
            version="v2",
            filters=filters,
            page_size=50
        )

        # Process each memory
        mood_data = []
        for memory in memories:
            mood_item = await process_mood_memory(memory, MOOD_SCORE_MAP)
            mood_data.append(mood_item)

        return {"mood_history_data": mood_data}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching mood history: {str(e)}"
        )

async def process_mood_memory(memory: Dict, mood_score_map: Dict) -> Dict:
    """Helper function to process individual mood memory"""
    mood_text = memory.get('memory', '')
    detected_mood = "neutral"

    if mood_text:
        try:
            parsed = await asyncio.to_thread(mood_llm.invoke, mood_text)
            mood_data = parse_mood_response(parsed)
            detected_mood = mood_data.get('mood', 'neutral').lower()
        except Exception as e:
            print(f"Error parsing mood from '{mood_text}': {str(e)}")
    
    timestamp = get_memory_timestamp(memory)
    
    return {
        "time": timestamp.isoformat(),
        "mood": detected_mood,
        "mood_score": mood_score_map.get(detected_mood, 3)
    }

def parse_mood_response(response) -> Dict:
    """Extract mood data from LLM response"""
    if isinstance(response, dict) and 'parsed' in response:
        response = response['parsed']
    if isinstance(response, PydanticBaseModel):
        return response.model_dump()
    return {}

def get_memory_timestamp(memory: Dict) -> datetime:
    """Extract and format timestamp from memory"""
    created_at = memory.get('created_at')
    if created_at:
        try:
            # Handle ISO format with timezone if present
            if 'Z' in created_at or '+' in created_at:
                return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            return datetime.fromisoformat(created_at.split('.')[0])
        except ValueError:
            pass
    return datetime.now()