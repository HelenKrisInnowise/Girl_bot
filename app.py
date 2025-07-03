import streamlit as st
import uuid
import os
from dotenv import load_dotenv
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# Import components from submodules
from modules.mem0_config import mem0_client, graph_mem0_client 
from modules.profiles import MAIN_CHARACTER_TRAITS, FORMALITY_LEVELS, COMMUNICATION_STYLES
from modules.llm_setup import generate_proactive_query, llm, mood_llm, get_system_prompt_template, generate_dynamic_profile, get_user_personal_profile, get_user_personal_profile_graph, generate_proactive_query_graph, detect_controversial_topic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel

# --- 0. Project Setup: Load Environment Variables ---
load_dotenv()

if not all([os.getenv("OPENAI_API_KEY"),os.getenv("MEM0_API_KEY"),
            os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")]):
    st.error("Please ensure all required environment variables are set in your .env file, including Neo4j credentials.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Girls Chatbot Demo (Neo4j Graph Memory)", layout="centered")
st.markdown(
    """
    <style>
    body { overflow-x: hidden; }
    .main .block-container {
        max-width: 800px; padding-top: 1rem; padding-right: 1rem; padding-left: 1rem; padding-bottom: 100px;
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

st.title("Girls Chatbot Demo (Neo4j Graph Memory)") # Updated title

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
    st.session_state.current_mood = None
    st.session_state.current_intent = None
    st.session_state.user_profile_summary = None
    st.session_state.user_profile_summary_graph = None
    st.session_state.mood_history_data = None
    st.session_state.proactive_query_suggestion = None
    st.session_state.suggested_topic_query = None # New state for suggested topic

    try:
        # Initial add to Mem0 will now use Graph Memory configured in modules/mem0_config.py
        mem0_client.add(messages=[
            {"role": "assistant", "content": "I am a chatbot designed to embody various girl personas. Use the sidebar to configure my personality!"}
        ], user_id=st.session_state.mem0_session_id)
        st.info("Initial memory added to Mem0 (with Graph Memory) for this session.")
    except Exception as e:
        st.warning(f"Could not add initial memory to Mem0 (Graph Memory): {e}")

# --- Dynamic Profile Selection UI in Sidebar ---
st.sidebar.header("Configure Chatbot Persona")

selected_traits = st.sidebar.multiselect(
    "1. Select Main Character Traits:",
    options=MAIN_CHARACTER_TRAITS,
    default=st.session_state.selected_traits,
    key="main_traits_selector"
)

selected_formality = st.sidebar.selectbox(
    "2. Select Formality Level:",
    options=FORMALITY_LEVELS,
    index=FORMALITY_LEVELS.index(st.session_state.selected_formality),
    key="formality_selector"
)

selected_style = st.sidebar.selectbox(
    "3. Select Communication Style:",
    options=COMMUNICATION_STYLES,
    index=COMMUNICATION_STYLES.index(st.session_state.selected_style),
    key="style_selector"
)

if st.sidebar.button("Generate Persona"):
    if not selected_traits and selected_formality == "Friendly" and selected_style == "Supportive":
        st.sidebar.warning("Please select at least one characteristic or change defaults to generate a persona.")
    else:
        with st.spinner("Generating new persona..."):
            new_profile = generate_dynamic_profile(selected_traits, selected_formality, selected_style)
            st.session_state.dynamic_profile = new_profile
            st.session_state.selected_traits = selected_traits
            st.session_state.selected_formality = selected_formality
            st.session_state.selected_style = selected_style
            st.session_state.messages = []
            st.session_state.mem0_session_id = str(uuid.uuid4())
            st.session_state.user_profile_summary = None
            st.session_state.user_profile_summary_graph = None
            st.session_state.mood_history_data = None
            st.session_state.proactive_query_suggestion = None
            st.session_state.proactive_query_suggestion_vector = None
            st.session_state.suggested_topic_query = None
            try:
                mem0_client.add(messages=[
                    {"role": "assistant", "content": f"Hello! I am now embodying a new persona: {st.session_state.dynamic_profile['description']}"}
                ], user_id=st.session_state.mem0_session_id)
                st.info("New persona generated and initial memory added to Mem0 (Graph Memory).")
            except Exception as e:
                st.warning(f"Could not add initial memory to Mem0 (Graph Memory) for new persona: {e}")
            st.sidebar.success("Persona updated! Conversation reset.")

st.sidebar.markdown("---")
st.sidebar.subheader("Current Chatbot Persona:")
st.sidebar.markdown(f"**Description:** {st.session_state.dynamic_profile['description']}")
st.sidebar.markdown(f"**Traits:** {st.session_state.dynamic_profile['behavioral_traits']}")
st.sidebar.markdown(f"**Selected Traits:** {', '.join(st.session_state.selected_traits) if st.session_state.selected_traits else 'None'}")
st.sidebar.markdown(f"**Formality:** {st.session_state.selected_formality}")
st.sidebar.markdown(f"**Style:** {st.session_state.selected_style}")

# --- User Personal Profile Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Personal Profile Vector")

if st.sidebar.button("Show/Update My Vector Profile"):
    with st.spinner("Fetching and summarizing your profile from memory..."):
        try:
            profile_filters = {
                "AND": [
                    {"user_id": st.session_state.mem0_session_id},
                    {"OR": [
                        {"categories": {"contains": "personal_details"}},
                        {"categories": {"contains": "user_interests"}},
                        {"categories": {"contains": "user_preferences"}},
                        {"categories": {"contains": "relationships"}},
                        {"categories": {"contains": "opinions"}}
                    ]}
                ]
            }
            personal_memories = mem0_client.get_all(version="v2", filters=profile_filters, page_size=30)
            
            st.session_state.user_profile_summary = get_user_personal_profile(personal_memories)
            st.sidebar.success("User profile updated!")
        except Exception as e:
            st.sidebar.error(f"Error fetching/summarizing profile: {e}")
            st.session_state.user_profile_summary = None

if st.session_state.user_profile_summary:
    profile = st.session_state.user_profile_summary
    st.sidebar.markdown(f"**Name:** {profile['name'] if profile['name'] else 'Not found'}")
    st.sidebar.markdown(f"**Summary:** {profile['summary']}")
else:
    st.sidebar.info("Click 'Show/Update My Profile' to generate your personal profile based on past conversations.")


# --- User Personal Profile Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Personal Profile Graph")

if st.sidebar.button("Show/Update My Graph Profile"):
    with st.spinner("Fetching and summarizing your profile from graph..."):
        try:
            # Calculate date some days ago for filtering
            # days_ago = datetime.now() - timedelta(days=50)
            # days_ago_iso = days_ago.isoformat(timespec='seconds') + 'Z' 
            # filters = {
            #     "AND": [
            #         {"user_id": st.session_state.mem0_session_id},
            #         {"created_at": {"gte": days_ago_iso}},
            #     ]
            # }
            graph_data = graph_mem0_client.get_all(user_id = st.session_state.mem0_session_id)
            st.session_state.user_profile_summary_graph = get_user_personal_profile_graph(graph_data)
            st.sidebar.success("User graph profile updated!")
        except Exception as e:
            st.sidebar.error(f"Error fetching/summarizing profile: {e}")
            st.session_state.user_profile_summary_graph = None

if st.session_state.user_profile_summary_graph:
    profile = st.session_state.user_profile_summary_graph
    st.sidebar.markdown(f"**Name:** {profile['name'] if profile['name'] else 'Not found'}")
    st.sidebar.markdown(f"**Summary:** {profile['summary']}")
else:
    st.sidebar.info("Click 'Show/Update My Profile' to generate your personal profile based on user's MemoryGraph.")
    
    
# --- Conversation Tools Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Conversation Tools")

if st.sidebar.button("Generate Proactive Query Vector"): # Renamed button
    with st.spinner("Thinking of a way to re-engage..."):
        try:
            # Calculate date 10 days ago for filtering
            ten_days_ago = datetime.now() - timedelta(days=10)

            ten_days_ago_iso = ten_days_ago.isoformat(timespec='seconds') + 'Z' 
            filters = {
                "AND": [
                    {"user_id": st.session_state.mem0_session_id},
                    {"created_at": {"gte": ten_days_ago_iso}},
                    {"OR": [
                        {"categories": {"contains": "life_events"}},
                        {"categories": {"contains": "daily_routine"}}
                    ]}
                ]
            }

            recent_relevant_memories = mem0_client.get_all(version="v2", filters=filters, page_size=15) # limit to 10 for prompt context
            proactive_query_vector = generate_proactive_query(recent_relevant_memories) # Call the renamed function
            st.session_state.proactive_query_suggestion_vector = proactive_query_vector # Store in session state
            st.sidebar.info(f"**Proactive Query Suggestion:** {proactive_query_vector}") # Display the suggestion
        except Exception as e:
            st.sidebar.error(f"Error generating proactive query: {e}")
            st.session_state.proactive_query_suggestion_vector = None

# Display the last proactive query suggestion
if st.session_state.proactive_query_suggestion:
    st.sidebar.markdown(f"**Proactive Query Vector:** {st.session_state.proactive_query_suggestion_vector}")
# --- 2. Generate Proactive Query on personal Graph memory (NEW button) ---
if st.sidebar.button("Generate Proactive Query (Graph Memory)"): # New button
    with st.spinner("Thinking of a proactive query from graph memory..."):
        try:
            graph_data = graph_mem0_client.get_all(user_id = st.session_state.mem0_session_id)
            proactive_query = generate_proactive_query_graph(graph_data)
            st.session_state.proactive_query_suggestion = proactive_query
            st.sidebar.info(f"**Proactive Query (Graph Memory):** {proactive_query}")
        except Exception as e:
            st.sidebar.error(f"Error generating proactive query from graph memory: {e}")
            st.session_state.proactive_query_suggestion = None

# Display the last proactive query suggestion
if st.session_state.proactive_query_suggestion:
    st.sidebar.markdown(f"**Proactive Query Vector:** {st.session_state.proactive_query_suggestion_vector}")
    st.sidebar.markdown(f"**Proactive Query Graph:** {st.session_state.proactive_query_suggestion}")

# --- Mood History Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("User Mood History")

if st.sidebar.button("Show Mood History"):
    with st.spinner("Fetching and processing mood history..."):
        try:
            month_ago = datetime.now() - timedelta(days=30)

            month_ago_iso = month_ago.isoformat(timespec='seconds') + 'Z' 
            filters = {
                "AND": [
                    {"user_id": st.session_state.mem0_session_id},
                    {"created_at": {"gte": month_ago_iso}},
                    {"categories": {"contains": "user_mood"}}
                ]
            }

            all_mood_memories = mem0_client.get_all(version="v2", filters=filters, page_size=50)
            # --- NEW: Use mood_llm to parse the mood from memory text ---
            mood_data_for_chart = []
            for memory in all_mood_memories:
                mood_text = memory.get('memory', '')
                detected_mood = "neutral" # Default if parsing fails

                if mood_text: # Only attempt to parse if there's text
                    try:
                        # Invoke mood_llm to parse the structured mood
                        parsed_mood_raw = mood_llm.invoke(mood_text)
                        
                        # Robustly extract Pydantic model and convert to dict
                        parsed_mood_data = {}
                        if isinstance(parsed_mood_raw, dict) and 'parsed' in parsed_mood_raw and isinstance(parsed_mood_raw['parsed'], BaseModel):
                            parsed_mood_data = parsed_mood_raw['parsed'].model_dump()
                        elif isinstance(parsed_mood_raw, BaseModel):
                            parsed_mood_data = parsed_mood_raw.model_dump()
                        
                        if parsed_mood_data and 'mood' in parsed_mood_data:
                            detected_mood = parsed_mood_data['mood'].lower()
                            # You could also use intensity if you want to refine score, e.g.,
                            # if parsed_mood_data.get('intensity') == 'high': score += 0.5
                            
                    except Exception as parse_e:
                        st.warning(f"Could not parse mood from memory '{mood_text}': {parse_e}. Defaulting to neutral.")
                        detected_mood = "neutral" # Fallback on parsing error

                created_at_iso = memory.get('created_at')
                if created_at_iso:
                    timestamp = datetime.fromisoformat(created_at_iso.split('.')[0]).replace(tzinfo=None)
                else:
                    timestamp = datetime.now()
                
                mood_data_for_chart.append({
                    "time": timestamp,
                    "mood": detected_mood,
                    "mood_score": MOOD_SCORE_MAP.get(detected_mood, 3)
                })
            
            if mood_data_for_chart:
                df_mood_history = pd.DataFrame(mood_data_for_chart).sort_values(by="time")
                st.session_state.mood_history_data = df_mood_history
                st.sidebar.success("Mood history fetched!")
            else:
                st.session_state.mood_history_data = None
                st.sidebar.info("No mood history found yet. Chat more to generate data!")

        except Exception as e:
            st.sidebar.error(f"Error fetching mood history: {e}")
            st.session_state.mood_history_data = None

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
else:
    if st.sidebar.button("How to get Mood History?"):
        st.sidebar.info("Chat with the bot, and your mood will be analyzed on each turn. Then click 'Show Mood History' to see the trend.")

# --- Display Chat History (from Streamlit session state) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
prompt = st.chat_input("Type your message here...")

if prompt:
    controversial_analysis_result = detect_controversial_topic(prompt)
    
    if controversial_analysis_result.get('is_controversial'):
        category = controversial_analysis_result.get('category', 'undefined')
        reason = controversial_analysis_result.get('reason', 'due to its content.')
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        refusal_message = f"I cannot discuss topics related to **{category}**, {reason}. Please let's move on to another topic."
        st.session_state.messages.append({"role": "assistant", "content": refusal_message})
        with st.chat_message("assistant"):
            st.markdown(refusal_message)
        
        try:
            mem0_client.add(messages=[{"role": "assistant", "content": f"Refused to discuss controversial topic: {category} for user message: '{prompt}'"}],
                            user_id=st.session_state.mem0_session_id,
                            categories=["chatbot_interactions"])
        except Exception as add_refusal_e:
            st.warning(f"Could not add refusal memory to Mem0 (Graph Memory): {add_refusal_e}")
        
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        mem0_client.add(messages=[{"role": "user", "content": prompt}], user_id=st.session_state.mem0_session_id)

    except Exception as e:
        st.warning(f"Could not add user message to Mem0 (Cloud Vector Database): {e}")

    try:
        # formatted_message = f"As user {st.session_state.mem0_session_id}, I share: {prompt}"
        graph_mem0_client.add(prompt, user_id=st.session_state.mem0_session_id)
    except Exception as e:
        st.warning(f"Could not add user message to Mem0 (Graph Memory): {e}")



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


    with st.chat_message("assistant"):
        with st.spinner(f"Thinking as a dynamically generated persona..."):
            profile_desc = st.session_state.dynamic_profile['description']
            profile_traits = st.session_state.dynamic_profile['behavioral_traits']

            system_prompt_template = get_system_prompt_template()

            system_message_content = system_prompt_template.format(
                persona_name="Dynamically Generated Persona",
                profile_description=profile_desc,
                profile_behavioral_traits=profile_traits,
                relevant_memories=relevant_memories_str,
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

