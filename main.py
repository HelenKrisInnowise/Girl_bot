from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict,  Annotated 
import asyncio
from datetime import datetime, timedelta
from fastapi import Depends 
from mem0 import AsyncMemoryClient, AsyncMemory  
from functools import wraps
import logging
import time
from fastapi.responses import StreamingResponse
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel as PydanticBaseModel
import warnings


from modules.mem0_config import create_mem0_clients

from modules.llm_setup import (
    generate_dynamic_profile, get_user_personal_profile, get_user_personal_profile_graph,
    generate_proactive_query, generate_proactive_query_graph,
    detect_controversial_topic, get_system_prompt_template, llm, mood_llm
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

# --- Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("FastAPI startup: Initializing Mem0 clients...")
    app.state.mem0_client, app.state.graph_mem0_client = await create_mem0_clients()
    print("FastAPI startup: Mem0 clients initialized.")
    yield


app = FastAPI(title="Girls Chatbot", lifespan=lifespan)

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

def log_execution_time(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Executing {func.__name__}")
        try:
            result = await func(*args, **kwargs)  # Note the await here
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return async_wrapper

# --- API Endpoints ---
@app.post("/generate_persona")
@log_execution_time
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
@log_execution_time
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


@app.post("/chat_stream")
@log_execution_time
async def chat_endpoint_streaming(
    request: ChatRequest,
    mem0_client: AsyncMemoryClient = Depends(get_mem0_client),
    graph_mem0_client: AsyncMemory = Depends(get_graph_client)
):
    
    async def generate():
        try:
            # Controversial topic detection
            start_time = time.time()
            controversial_analysis_result = await detect_controversial_topic(request.prompt)
            if controversial_analysis_result.get('is_controversial'):
                category = controversial_analysis_result.get('category', 'undefined')
                refusal_message = controversial_analysis_result.get('refusal_message', 'due to its content.')
                refusal_message = f"{refusal_message}"
                
                yield json.dumps({
                    "response": refusal_message,
                        "messages": [msg.model_dump() for msg in request.messages] + [
                        {"role": "assistant", "content": refusal_message}
                    ],
                    "relevant_memories_str": "N/A"
                }) + "\n"
                print(f"Controversial topic detection took {time.time() - start_time:.2f}s")
                return
            
            print(f"Controversial topic detection took {time.time() - start_time:.2f}s")
            
            # --- Start operations with Mem0 in parallel ---
            # Start adding memory in the background. Its result is not needed immediately.
            add_task = asyncio.create_task(add_memory_to_mem0(mem0_client, request.prompt, request.user_id))
            
            graph_add_task = asyncio.create_task(add_memory_to_graph(graph_mem0_client, request.prompt, request.user_id))
            # Start searching for relevant memories. The result will be needed for system_message.
            search_task = asyncio.create_task(search_mem0_for_relevant_memories(mem0_client, request.prompt, request.user_id))

            # Wait for the memory search results, as they are needed for system_message
            relevant_memories_str = await search_task 
            
            # Now that we have relevant_memories_str, we can prepare system_message
            system_message_content = get_system_prompt_template().format(
                persona_name="Dynamically Generated Persona",
                profile_description=request.dynamic_profile['description'],
                profile_behavioral_traits=request.dynamic_profile['behavioral_traits'],
                relevant_memories=relevant_memories_str,
            )

            # Prepare messages for LLM
            llm_messages = [SystemMessage(content=system_message_content)]
            for msg in request.messages:
                llm_messages.append(
                    HumanMessage(content=msg.content) if msg.role == "user" 
                    else AIMessage(content=msg.content)
                )
            llm_messages.append(HumanMessage(content=request.prompt))

            # Stream the response
            full_response = ""
            async for chunk in llm.astream(llm_messages):
                content = chunk.content
                full_response += content
                yield json.dumps({
                    "response": content,
                        "messages": [msg.model_dump() for msg in request.messages] + [
                        {"role": "assistant", "content": full_response}
                    ],
                    "relevant_memories_str": relevant_memories_str
                }) + "\n"

           # Wait for the background task to add memory to complete
            await add_task, graph_add_task 

        except Exception as e:
            yield json.dumps({
                "error": str(e),
                "status_code": 500
            }) + "\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# Helper functions to encapsulate Mem0 logic
async def add_memory_to_mem0(client: AsyncMemoryClient, prompt: str, user_id: str):
    add_start_time = time.time()
    try:
        await client.add(
            messages=[{"role": "user", "content": prompt}],
            user_id=user_id
        )
        print(f"Adding memory took {time.time() - add_start_time:.2f}s")
    except Exception as e:
        print(f"Could not add user message to Mem0 (Cloud Vector Database): {e}")
        
async def add_memory_to_graph(client: AsyncMemory, prompt: str, user_id: str):
    add_start_time = time.time()
    try:
        await client.add(prompt,user_id=user_id
        )
        print(f"Adding Graph memory took {time.time() - add_start_time:.2f}s")
    except Exception as e:
        print(f"Could not add user message to Mem0 Graph: {e}")
        
async def search_mem0_for_relevant_memories(client: AsyncMemoryClient, query: str, user_id: str) -> str:
    relevant_memories_str = "No relevant memories found."
    search_start_time = time.time()
    try:
        search_results = await client.search(
            query=query,
            user_id=user_id,
            limit=3
        )
        if search_results:
            relevant_memories_str = "\n".join([f"- {m['memory']}" for m in search_results])
        print(f"searching memory took {time.time() - search_start_time:.2f}s")    
    except Exception as e:
        print(f"Search error: {e}")
    return relevant_memories_str


@app.post("/get_user_profile_vector")
@log_execution_time
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
        
@app.post("/generate_proactive_query_vector")
@log_execution_time
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

@app.post("/get_mood_history")
@log_execution_time
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