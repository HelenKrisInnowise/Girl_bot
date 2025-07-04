import streamlit as st
import uuid
import os
from dotenv import load_dotenv
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import httpx 
import json
from pydantic import BaseModel 
from typing import Literal
from modules.profiles import MAIN_CHARACTER_TRAITS, FORMALITY_LEVELS, COMMUNICATION_STYLES

class Message(BaseModel):
    role: str
    content: str

load_dotenv()

FASTAPI_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

# Basic check for essential environment variables
if not all([os.getenv("OPENAI_API_KEY"), os.getenv("MEM0_API_KEY")]):
    st.error("Please ensure OPENAI_API_KEY and MEM0_API_KEY are set in your .env file.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Girls Chatbot Demo (FastAPI Backend)", layout="centered")
st.markdown(
    """
    <style>
    body { overflow-x: hidden; }
    .main .block-container {
        max_width: 800px; padding-top: 1rem; padding-right: 1rem; padding-left: 1rem; padding-bottom: 100px;
    }
    .stChatInput {
        position: fixed; bottom: 0; left: 165px; width: calc(100% - 165px); padding: 10px;
        background-color: white; border-top: 1px solid #f0f2f6; box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        z-index: 1000; box-sizing: border-box;
    }
    .stChatMessage {
        border-radius: 15px; padding: 10px 15px; margin-bottom: 10px; max-width: 70%;
        word-wrap: break-word; color: black; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stChatMessage.stChatMessage--user {
        background-color: #e0f7fa; align-self: flex-end; margin-left: auto; border-bottom-right-radius: 2px;
    }
    .stChatMessage.stChatMessage--assistant {
        background-color: #fce4ec; align-self: flex-start; margin-right: auto; border-bottom-left-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Girls Chatbot Demo (FastAPI Backend)")

# --- Mood Mapping for Visualization ---
MOOD_SCORE_MAP = {
    "joyful": 5, "excited": 4, "neutral": 3, "confused": 2.5, "surprised": 3.5,
    "fearful": 1.5, "anxious": 1.7, "sad": 1, "angry": 0.5, "disgusted": 0
}

# --- Streamlit Session State Initialization ---
if "mem0_session_id" not in st.session_state:
    st.session_state.mem0_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.selected_traits = []
    st.session_state.selected_formality = "Friendly"
    st.session_state.selected_style = "Supportive"
    st.session_state.dynamic_profile = {
        "description": "A versatile chatbot waiting for your personality settings.",
        "behavioral_traits": "The chatbot will be neutral until specific traits are selected."
    }
    st.session_state.user_profile_summary = None
    st.session_state.user_profile_summary_graph = None
    st.session_state.mood_history_data = None
    st.session_state.proactive_query_suggestion_vector = None
    st.session_state.proactive_query_suggestion_graph = None
    st.session_state.chat_mode = "streaming"

    # Call backend to add initial memory
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{FASTAPI_BACKEND_URL}/add_initial_memory", 
                                 params={"user_id": st.session_state.mem0_session_id, 
                                        "description": st.session_state.dynamic_profile['description']})
            response.raise_for_status()
            st.info("Initial memory added to Mem0 for this session.")
    except httpx.HTTPStatusError as e:
        st.warning(f"Could not add initial memory: {e.response.text}")
    except Exception as e:
        st.warning(f"Could not add initial memory: {e}")

# --- Dynamic Profile Selection UI in Sidebar ---
st.sidebar.header("Configure Chatbot Persona")

selected_traits = st.sidebar.multiselect(
    "1. Select Main Character Traits:",
    options=MAIN_CHARACTER_TRAITS,
    default=st.session_state.get('selected_traits', []),
    key="main_traits_selector"
)

selected_formality = st.sidebar.selectbox(
    "2. Select Formality Level:",
    options=FORMALITY_LEVELS,
    index=FORMALITY_LEVELS.index(st.session_state.get('selected_formality', "Friendly")),
    key="formality_selector"
)

selected_style = st.sidebar.selectbox(
    "3. Select Communication Style:",
    options=COMMUNICATION_STYLES,
    index=COMMUNICATION_STYLES.index(st.session_state.get('selected_style', "Supportive")),
    key="style_selector"
)

if st.sidebar.button("Generate Persona"):
    if not selected_traits and selected_formality == "Friendly" and selected_style == "Supportive":
        st.sidebar.warning("Please select at least one characteristic or change defaults.")
    else:
        with st.spinner("Generating new persona..."):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(f"{FASTAPI_BACKEND_URL}/generate_persona", json={
                        "traits": selected_traits,
                        "formality": selected_formality,
                        "style": selected_style
                    })
                    response.raise_for_status()
                    new_profile = response.json()["dynamic_profile"]
                
                st.session_state.dynamic_profile = new_profile
                st.session_state.selected_traits = selected_traits
                st.session_state.selected_formality = selected_formality
                st.session_state.selected_style = selected_style
                st.session_state.messages = []
                st.session_state.mem0_session_id = str(uuid.uuid4())
                st.session_state.user_profile_summary = None
                st.session_state.user_profile_summary_graph = None
                st.session_state.mood_history_data = None
                st.session_state.proactive_query_suggestion_vector = None


                # Add initial memory for new persona
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(f"{FASTAPI_BACKEND_URL}/add_initial_memory", 
                                         params={"user_id": st.session_state.mem0_session_id, 
                                                "description": st.session_state.dynamic_profile['description']})
                    response.raise_for_status()
                st.sidebar.success("Persona updated! Conversation reset.")
                st.rerun()
            except httpx.HTTPStatusError as e:
                st.sidebar.error(f"Error generating persona: {e.response.text}")
            except Exception as e:
                st.sidebar.error(f"Error generating persona: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Current Chatbot Persona:")
st.sidebar.markdown(f"**Description:** {st.session_state.dynamic_profile['description']}")
st.sidebar.markdown(f"**Traits:** {st.session_state.dynamic_profile['behavioral_traits']}")
st.sidebar.markdown(f"**Selected Traits:** {', '.join(st.session_state.selected_traits) if st.session_state.selected_traits else 'None'}")
st.sidebar.markdown(f"**Formality:** {st.session_state.selected_formality}")
st.sidebar.markdown(f"**Style:** {st.session_state.selected_style}")

# --- Chat Mode Selection ---
st.sidebar.markdown("---")
st.sidebar.subheader("Chat Mode")
chat_mode = st.sidebar.radio(
    "Select chat mode:",
    ["Regular", "Streaming"],
    index=0 if st.session_state.chat_mode == "regular" else 1,
    key="chat_mode_selector"
)
st.session_state.chat_mode = "regular" if chat_mode == "Regular" else "streaming"

# --- User Profile Sections ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Personal Profiles")

if st.sidebar.button("Show/Update My Vector Profile"):
    with st.spinner("Fetching vector profile..."):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{FASTAPI_BACKEND_URL}/get_user_profile_vector", 
                                     params={"user_id": st.session_state.mem0_session_id})
                response.raise_for_status()
                st.session_state.user_profile_summary = response.json()["profile_summary"]
            st.sidebar.success("User vector profile updated!")
        except httpx.HTTPStatusError as e:
            st.sidebar.error(f"Error fetching vector profile: {e.response.text}")
        except Exception as e:
            st.sidebar.error(f"Error fetching vector profile: {e}")

if st.session_state.user_profile_summary:
    profile = st.session_state.user_profile_summary
    st.sidebar.markdown(f"**Name (Vector):** {profile['name'] if profile['name'] else 'Not found'}")
    st.sidebar.markdown(f"**Summary (Vector):** {profile['summary']}")
else:
    st.sidebar.info("Click above to generate your personal profile")

if st.sidebar.button("Generate Proactive Query (Vector)"):
    with st.spinner("Thinking of a proactive query..."):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{FASTAPI_BACKEND_URL}/generate_proactive_query_vector", 
                                     params={"user_id": st.session_state.mem0_session_id})
                response.raise_for_status()
                st.session_state.proactive_query_suggestion_vector = response.json()["proactive_query"]
            st.sidebar.info(f"**Proactive Query (Vector):** {st.session_state.proactive_query_suggestion_vector}")
        except httpx.HTTPStatusError as e:
            st.sidebar.error(f"Error generating query: {e.response.text}")
        except Exception as e:
            st.sidebar.error(f"Error generating query: {e}")

# --- Mood History Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Mood History")

if st.sidebar.button("Show Mood History"):
    with st.spinner("Fetching mood history..."):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{FASTAPI_BACKEND_URL}/get_mood_history", 
                                     params={"user_id": st.session_state.mem0_session_id})
                response.raise_for_status()
                mood_history_data_raw = response.json()["mood_history_data"]
            
            if mood_history_data_raw:
                df_mood_history = pd.DataFrame(mood_history_data_raw)
                df_mood_history['time'] = pd.to_datetime(df_mood_history['time'])
                df_mood_history = df_mood_history.sort_values(by="time")
                st.session_state.mood_history_data = df_mood_history
                st.sidebar.success("Mood history fetched!")
            else:
                st.session_state.mood_history_data = None
                st.sidebar.info("No mood history found yet. Chat more to generate data!")
        except httpx.HTTPStatusError as e:
            st.sidebar.error(f"Error fetching mood history: {e.response.text}")
        except Exception as e:
            st.sidebar.error(f"Error fetching mood history: {e}")

if st.session_state.mood_history_data is not None and not st.session_state.mood_history_data.empty:
    st.sidebar.markdown("#### Mood Dynamics Over Time")
    
    chart = alt.Chart(st.session_state.mood_history_data).mark_line(point=True).encode(
        x=alt.X('time', axis=alt.Axis(title='Time', format='%H:%M')),
        y=alt.Y('mood_score', axis=alt.Axis(title='Mood Score', values=list(MOOD_SCORE_MAP.values()),
                                          labelExpr="datum.value == 0.5 ? 'Angry' : datum.value == 1 ? 'Sad' : datum.value == 1.5 ? 'Fearful' : datum.value == 1.7 ? 'Anxious' : datum.value == 2.5 ? 'Confused' : datum.value == 3 ? 'Neutral' : datum.value == 3.5 ? 'Surprised' : datum.value == 4 ? 'Excited' : datum.value == 5 ? 'Joyful' : ''")),
        tooltip=['time', 'mood', 'mood_score']
    ).properties(
        title='User Mood Trend'
    ).interactive()
    
    st.sidebar.altair_chart(chart, use_container_width=True)

# --- Display Chat History ---
for message in st.session_state.messages:
    st.chat_message(message.role).markdown(message.content)

# --- Chat Input and Logic ---
prompt = st.chat_input("Type your message here...")

# Modify the streaming chat section in your Streamlit app:

# Modify the streaming chat section in your Streamlit app:

if prompt:
    st.session_state.messages.append(Message(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_payload = {
        "user_id": st.session_state.mem0_session_id,
        "prompt": prompt,
        "messages": [msg.dict() for msg in st.session_state.messages],
        "selected_traits": st.session_state.selected_traits,
        "selected_formality": st.session_state.selected_formality,
        "selected_style": st.session_state.selected_style,
        "dynamic_profile": st.session_state.dynamic_profile
    }

    if st.session_state.chat_mode == "regular":
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(f"{FASTAPI_BACKEND_URL}/chat", json=chat_payload)
                response.raise_for_status()
                chat_response_data = response.json()
            
            st.session_state.messages = [Message(**msg) for msg in chat_response_data["messages"]]
            
            if chat_response_data.get("relevant_memories_str"):
                st.sidebar.subheader("Mem0 Search Results:")
                st.sidebar.markdown(chat_response_data["relevant_memories_str"])
            
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                with httpx.Client(timeout=180.0) as client:
                    with client.stream(
                        "POST",
                        f"{FASTAPI_BACKEND_URL}/chat_stream",
                        json=chat_payload
                    ) as response:
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line)
                                if "error" in data:
                                    raise Exception(data["error"])
                                
                                chunk = data["response"]
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                                
                                # Update messages with the latest response
                                st.session_state.messages = [
                                    Message(**msg) for msg in data["messages"]
                                ]
                
                message_placeholder.markdown(full_response)
                
                if data.get("relevant_memories_str"):
                    st.sidebar.subheader("Mem0 Search Results:")
                    st.sidebar.markdown(data["relevant_memories_str"])
                
            except Exception as e:
                st.error(f"Error during streaming: {str(e)}")