import streamlit as st
import uuid
import os
from dotenv import load_dotenv
import pandas as pd 
import altair as alt 
from datetime import datetime


from modules.mem0_config import mem0_client
from modules.profiles import MAIN_CHARACTER_TRAITS, FORMALITY_LEVELS, COMMUNICATION_STYLES 
from modules.pydantic_models import MoodAttributes, IntentAttributes 
from modules.llm_setup import llm, mood_llm, intent_llm, get_system_prompt_template, generate_dynamic_profile, DynamicProfileOutput, get_user_personal_profile, suggest_conversation_topic # Import DynamicProfileOutput from llm_setup
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel


load_dotenv()

if not all([os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_ENDPOINT"),
            os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME"), os.getenv("MEM0_API_KEY")]):
    st.error("Please ensure all required environment variables are set in your .env file.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Girls Chatbot Demo", layout="centered")
st.markdown(
    """
    <style>
    /* Ensure no horizontal scrollbar */
    body {
        overflow-x: hidden;
    }
    .reportview-container {
        flex-direction: row-reverse;
    }
    .main .block-container {
        /* Keep max-width for content, but adjust padding for chat input */
        max-width: 800px;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 100px; /* Ensure space for the fixed chat input */
    }
    /* Simplified chat input styling to ensure visibility */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 165px; /* Adjusted: pushes the input further to the left */
        width: calc(100% - 165px); /* Adjusted: fills the remaining space */
        padding: 10px;
        background-color: white;
        border-top: 1px solid #f0f2f6;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        z-index: 1000;
        box-sizing: border-box; /* Include padding in width */
    }
    /* Custom styling for chat bubbles */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 70%;
        word-wrap: break-word;
        color: black;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stChatMessage.stChatMessage--user {
        background-color: #e0f7fa;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    .stChatMessage.stChatMessage--assistant {
        background-color: #fce4ec;
        align-self: flex-start;
        margin-right: auto;
        border-bottom-left-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Girls Chatbot Demo")

# --- Mood Mapping for Visualization ---
# Define a mapping from mood strings to numerical values for plotting
MOOD_SCORE_MAP = {
    "joyful": 5,
    "excited": 4,
    "neutral": 3,
    "confused": 2.5,
    "surprised": 3.5,
    "fearful": 1.5,
    "anxious": 1.7,
    "sad": 1,
    "angry": 0.5,
    "disgusted": 0
}

# --- Streamlit Session State Initialization ---
if "mem0_session_id" not in st.session_state:
    st.session_state.mem0_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    # Initialize dynamic profile selection states
    st.session_state.selected_traits = []
    st.session_state.selected_formality = "Friendly" # Default
    st.session_state.selected_style = "Supportive" # Default
    st.session_state.dynamic_profile = { # Default dynamic profile
        "description": "A versatile chatbot waiting for your personality settings.",
        "behavioral_traits": "The chatbot will be neutral until specific traits are selected."
    }

    st.session_state.current_mood = None
    st.session_state.current_intent = None
    st.session_state.user_profile_summary = None # Initialize user profile summary
    st.session_state.mood_history_data = None # Initialize for mood history chart

    try:
        mem0_client.add(messages=[
            {"role": "assistant", "content": "I am a chatbot designed to embody various girl personas. Use the sidebar to configure my personality!"}
        ], user_id=st.session_state.mem0_session_id)
        st.info("Initial memory added to Mem0 for this session.")
    except Exception as e:
        st.warning(f"Could not add initial memory to Mem0: {e}")

# --- Dynamic Profile Selection UI in Sidebar ---
st.sidebar.header("Configure Chatbot Persona")

# 1. Selection of main character traits
selected_traits = st.sidebar.multiselect(
    "1. Select Main Character Traits:",
    options=MAIN_CHARACTER_TRAITS, # Using list from modules/profiles.py
    default=st.session_state.selected_traits,
    key="main_traits_selector"
)

# 2. Setting the level of formality of communication
selected_formality = st.sidebar.selectbox(
    "2. Select Formality Level:",
    options=FORMALITY_LEVELS, # Using list from modules/profiles.py
    index=FORMALITY_LEVELS.index(st.session_state.selected_formality),
    key="formality_selector"
)

# 3. Selection of communication style
selected_style = st.sidebar.selectbox(
    "3. Select Communication Style:",
    options=COMMUNICATION_STYLES, # Using list from modules/profiles.py
    index=COMMUNICATION_STYLES.index(st.session_state.selected_style),
    key="style_selector"
)

# Use a button to trigger regeneration explicitly to avoid too many LLM calls
if st.sidebar.button("Generate Persona"):
    # Check if a selection has been made, or if it's the default and no traits are selected
    if not selected_traits and selected_formality == "Friendly" and selected_style == "Supportive":
        st.sidebar.warning("Please select at least one characteristic or change defaults to generate a persona.")
    else:
        with st.spinner("Generating new persona..."):
            new_profile = generate_dynamic_profile(selected_traits, selected_formality, selected_style)
            st.session_state.dynamic_profile = new_profile
            st.session_state.selected_traits = selected_traits
            st.session_state.selected_formality = selected_formality
            st.session_state.selected_style = selected_style
            st.session_state.messages = [] # Clear local chat history for new persona
            st.session_state.mem0_session_id = str(uuid.uuid4()) # Start new mem0 session
            st.session_state.user_profile_summary = None # Reset user profile summary on new persona
            st.session_state.mood_history_data = None # Reset mood history on new persona
            try:
                mem0_client.add(messages=[
                    {"role": "assistant", "content": f"Hello! I am now embodying a new persona: {st.session_state.dynamic_profile['description']}"}
                ], user_id=st.session_state.mem0_session_id)
                st.info("New persona generated and initial memory added to Mem0.")
            except Exception as e:
                st.warning(f"Could not add initial memory to Mem0 for new persona: {e}")
            st.sidebar.success("Persona updated! Conversation reset.")

# Display current dynamic profile in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Current Chatbot Persona:")
st.sidebar.markdown(f"**Description:** {st.session_state.dynamic_profile['description']}")
st.sidebar.markdown(f"**Traits:** {st.session_state.dynamic_profile['behavioral_traits']}")
st.sidebar.markdown(f"**Selected Traits:** {', '.join(st.session_state.selected_traits) if st.session_state.selected_traits else 'None'}")
st.sidebar.markdown(f"**Formality:** {st.session_state.selected_formality}")
st.sidebar.markdown(f"**Style:** {st.session_state.selected_style}")


# --- User Personal Profile Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Personal Profile")

if st.sidebar.button("Show/Update My Profile"):
    with st.spinner("Fetching and summarizing your profile from memory..."):
        try:
            # Search Mem0 for personal details and interests
            personal_memories = mem0_client.search(
                query="personal information about the user, user's interests, hobbies, and preferences",
                user_id=st.session_state.mem0_session_id,
                # Filter by custom categories
                categories=["personal_details", "user_interests", "user_preferences"],
                limit=10 # Get enough relevant memories
            )
            # Use LLM to summarize these memories into a structured profile
            st.session_state.user_profile_summary = get_user_personal_profile(personal_memories)
            st.sidebar.success("User profile updated!")
        except Exception as e:
            st.sidebar.error(f"Error fetching/summarizing profile: {e}")
            st.session_state.user_profile_summary = None # Clear on error

if st.session_state.user_profile_summary:
    profile = st.session_state.user_profile_summary
    st.sidebar.markdown(f"**Name:** {profile['name'] if profile['name'] else 'Not found'}")
    st.sidebar.markdown(f"**Interests:** {', '.join(profile['interests']) if profile['interests'] else 'None'}")
    st.sidebar.markdown(f"**Preferences:** {', '.join(profile['preferences']) if profile['preferences'] else 'None'}")
    st.sidebar.markdown(f"**Summary:** {profile['summary']}")
else:
    st.sidebar.info("Click 'Show/Update My Profile' to generate your personal profile based on past conversations.")


# --- Suggest a Topic Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Conversation Topics")

if st.sidebar.button("Suggest a Topic"):
    with st.spinner("Thinking of a topic based on our past conversations..."):
        try:
            # Search Mem0 for preferred conversation topics
            topic_memories = mem0_client.search(
                query="suggest a conversation topic based on my preferred topics",
                user_id=st.session_state.mem0_session_id,
                categories=["preferred_conversation_topics"],
                limit=5
            )
            # Use LLM to suggest a topic
            suggested_topic = suggest_conversation_topic(topic_memories)
            st.sidebar.info(f"**Suggested Topic:** {suggested_topic}")
        except Exception as e:
            st.sidebar.error(f"Error suggesting topic: {e}")

# --- Mood History Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Mood History")

if st.sidebar.button("Show Mood History"):
    with st.spinner("Fetching and processing mood history..."):
        try:
            # Retrieve all user_mood memories
            # Using get_all for comprehensive history, or search if you want to filter/limit
            all_mood_memories = mem0_client.search(
                query="user's emotional state or mood changes",
                user_id=st.session_state.mem0_session_id,
                categories=["user_mood"],
                limit=50, # Limit to a reasonable number for the chart
                # Mem0 search results are usually ordered by relevance, you might need to sort by timestamp if Mem0 adds it.
            )
            
            # Prepare data for plotting
            mood_data_for_chart = []
            for memory in all_mood_memories:
                # The 'memory' field is expected to be a concise summary like "joyful, medium intensity"
                # We need to parse it to extract the mood and optionally intensity.
                # Example: "Mood: joyful, Intensity: medium, Reason: good news"
                mood_text = memory.get('memory', '')
                detected_mood = "neutral" # Default if not parsed
                
                # Simple parsing logic for the mood string
                # This could be more robust with regex if the format varies
                if "mood:" in mood_text.lower():
                    # Attempt to extract mood from a structured part
                    parts = mood_text.split("Mood:")
                    if len(parts) > 1:
                        mood_part = parts[1].split(',')[0].strip().lower()
                        # Clean up and map
                        for mood_key in MOOD_SCORE_MAP.keys():
                            if mood_key in mood_part:
                                detected_mood = mood_key
                                break
                else:
                    # Fallback for simpler mood mentions
                    for mood_key in MOOD_SCORE_MAP.keys():
                        if mood_key in mood_text.lower():
                            detected_mood = mood_key
                            break

                # Get timestamp and convert to datetime object
                created_at_iso = memory.get('created_at')
                timestamp = datetime.fromisoformat(created_at_iso.replace('Z', '+00:00')) if created_at_iso else datetime.now()
                
                mood_data_for_chart.append({
                    "time": timestamp,
                    "mood": detected_mood,
                    "mood_score": MOOD_SCORE_MAP.get(detected_mood, 3) # Default to neutral if not found
                })
            
            if mood_data_for_chart:
                # Create DataFrame and sort by time to ensure correct plotting order
                df_mood_history = pd.DataFrame(mood_data_for_chart).sort_values(by="time")
                st.session_state.mood_history_data = df_mood_history
                st.sidebar.success("Mood history fetched!")
            else:
                st.session_state.mood_history_data = None
                st.sidebar.info("No mood history found yet. Chat more to generate data!")

        except Exception as e:
            st.sidebar.error(f"Error fetching mood history: {e}")
            st.session_state.mood_history_data = None

# Display mood history chart if data exists
if st.session_state.mood_history_data is not None and not st.session_state.mood_history_data.empty:
    st.sidebar.markdown("#### Mood Dynamics Over Time")
    
    # Create an Altair chart
    chart = alt.Chart(st.session_state.mood_history_data).mark_line(point=True).encode(
        x=alt.X('time', axis=alt.Axis(title='Time', format='%H:%M')),
        y=alt.Y('mood_score', axis=alt.Axis(title='Mood Score', values=list(MOOD_SCORE_MAP.values()),
                                            labelExpr="datum.value == 0.5 ? 'Angry' : datum.value == 1 ? 'Sad' : datum.value == 1.5 ? 'Fearful' : datum.value == 1.7 ? 'Anxious' : datum.value == 2.5 ? 'Confused' : datum.value == 3 ? 'Neutral' : datum.value == 3.5 ? 'Surprised' : datum.value == 4 ? 'Excited' : datum.value == 5 ? 'Joyful' : ''")),
        tooltip=['time', 'mood', 'mood_score']
    ).properties(
        title='User Mood Trend'
    ).interactive() # Make the chart interactive (zoom, pan)
    
    st.sidebar.altair_chart(chart, use_container_width=True)
else:
    if st.sidebar.button("How to get Mood History?"):
        st.sidebar.info("Chat with the bot, and your mood will be analyzed on each turn. Then click 'Show Mood History' to see the trend. Ensure your API key has necessary permissions for Mem0.")


# --- Display Chat History (from Streamlit session state) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
prompt = st.chat_input("Type your message here...")

if prompt: # Only proceed if there's a prompt
    # Add user message to Streamlit history (local chat history)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message immediately (this is important for interactive feel)
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Add user message to Mem0 for searchability ---
    try:
        mem0_client.add(messages=[{"role": "user", "content": prompt}], user_id=st.session_state.mem0_session_id)
    except Exception as e:
        st.warning(f"Could not add user message to Mem0: {e}")


    # --- Retrieve relevant memories from Mem0 using search ---
    relevant_memories_str = "No relevant memories found."
    try:
        search_results = mem0_client.search(query=prompt, user_id=st.session_state.mem0_session_id, limit=3)
        if search_results:
            relevant_memories_str = "\n".join([f"- {m['memory']}" for m in search_results])
            st.sidebar.subheader("Mem0 Search Results:")
            for i, memory in enumerate(search_results):
                st.sidebar.markdown(f"**{i+1}.** {memory['memory']}")
        else:
            st.sidebar.subheader("Mem0 Search Results:")
            st.sidebar.info("No relevant memories found in Mem0 for this query.")
    except Exception as e:
        st.warning(f"Could not perform Mem0 search: {e}. Proceeding without additional memories.")

    # --- Mood and Intent Detection ---
    user_mood_str = "Not detected."
    user_intent_str = "Not detected."
    try:
        with st.spinner("Analyzing your mood and intent..."):
            mood_analysis_raw = mood_llm.invoke(prompt)
            intent_analysis_raw = intent_llm.invoke(prompt)

            current_mood_data = {}
            current_intent_data = {}

            if isinstance(mood_analysis_raw, dict) and 'parsed' in mood_analysis_raw and isinstance(mood_analysis_raw['parsed'], BaseModel):
                current_mood_data = mood_analysis_raw['parsed'].model_dump()
            elif isinstance(mood_analysis_raw, BaseModel):
                current_mood_data = mood_analysis_raw.model_dump()
            else:
                st.warning(f"Mood analysis result not a recognizable Pydantic model or dict with 'parsed'. Type: {type(mood_analysis_raw)}")

            if isinstance(intent_analysis_raw, dict) and 'parsed' in intent_analysis_raw and isinstance(intent_analysis_raw['parsed'], BaseModel):
                current_intent_data = intent_analysis_raw['parsed'].model_dump()
            elif isinstance(intent_analysis_raw, BaseModel):
                current_intent_data = intent_analysis_raw.model_dump()
            else:
                st.warning(f"Intent analysis result not a recognizable Pydantic model or dict with 'parsed'. Type: {type(intent_analysis_raw)}")


            st.session_state.current_mood = current_mood_data
            st.session_state.current_intent = current_intent_data

            if current_mood_data:
                user_mood_str = f"Mood: {current_mood_data.get('mood', 'unknown')}, Intensity: {current_mood_data.get('intensity', 'unknown')}"
                if current_mood_data.get('reason'):
                    user_mood_str += f", Reason: {current_mood_data['reason']}"
            
            # Add the detected mood to Mem0 specifically, tagged with 'user_mood'
            if user_mood_str != "Not detected.":
                try:
                    mem0_client.add(messages=[{"role": "user", "content": f"User's mood detected: {user_mood_str}"}],
                                    user_id=st.session_state.mem0_session_id,
                                    categories=["user_mood"]) # Explicitly tag with user_mood
                except Exception as add_mood_e:
                    st.warning(f"Could not add detected mood to Mem0: {add_mood_e}")

            if current_intent_data:
                user_intent_str = f"{current_intent_data.get('intent', 'unknown')}"
                if current_intent_data.get('target'):
                    user_intent_str += f" - target: {current_intent_data['target']}"
                if current_intent_data.get('details'):
                    user_intent_str += f" - details: {current_intent_data['details']}"

            st.sidebar.subheader("User Analysis:")
            st.sidebar.markdown(f"**Mood:** {user_mood_str}")
            st.sidebar.markdown(f"**Intent:** {user_intent_str}")

    except Exception as e:
        st.warning(f"Could not analyze mood/intent: {e}. Proceeding without it.")


    # --- Adaptive Response Generation ---
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as a dynamically generated persona..."):
            # Use dynamic_profile from session state
            profile_desc = st.session_state.dynamic_profile['description']
            profile_traits = st.session_state.dynamic_profile['behavioral_traits']

            system_prompt_template = get_system_prompt_template()

            system_message_content = system_prompt_template.format(
                persona_name="Dynamically Generated Persona", # Generic name as it's dynamic
                profile_description=profile_desc,
                profile_behavioral_traits=profile_traits,
                relevant_memories=relevant_memories_str,
                user_mood=user_mood_str,
                user_intent=user_intent_str
            )

            messages = [
                SystemMessage(content=system_message_content)
            ]

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            ai_response = llm.invoke(messages)
            st.markdown(ai_response.content)

            st.session_state.messages.append({"role": "assistant", "content": ai_response.content})
