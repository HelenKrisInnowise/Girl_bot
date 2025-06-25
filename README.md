```markdown
# Girls Chatbot Demo Project!

This project is a demonstration chatbot application built with Streamlit. The chatbot uses Azure DeepAI models via LangChain for response generation, manages conversation context with Mem0, and supports dynamically generated profiles for users, as well as analyzing user mood and intentions to create a more friendly and supportive interaction.

## Project Features
- **Dynamic Persona Generation:** You can customize the chatbot's personality by selecting main character traits, formality level, and communication style via the sidebar.

- **Mood and Intent Analysis:** The bot analyzes your messages to determine your mood and intentions, adapting its responses for more empathetic interactions.

- **Memory Management with Mem0:** Uses Mem0 to store key information from conversations, such as personal user data, interests, preferences, and mood history. This allows the bot to "remember" you and use this information for contextual responses.

- **User Profile:** Ability to generate/update a summarized user profile based on stored memories.

- **Conversation Topic Suggestions:** Feature that suggests discussion topics based on your previous preferences.

- **Mood Change History:** Visualization of your mood changes over time on a graph.

## Prerequisites
To run the project, you will need:

- Python 3.11+

- uv: A tool for managing Python packages. If you don't have it, install with:

```bash
pip install uv
```

## Project Installation
Clone the repository (if applicable):

```bash
git clone <YOUR_REPOSITORY_URL>
cd girl-bot
```

(If you're already in the `girl-bot` folder, skip this step)

Install dependencies using uv:
Navigate to the root directory where `pyproject.toml` is located and run:

```bash
uv pip install -e .
```

This will install all required dependencies listed in `pyproject.toml`, including `mem0ai`, `langchain`, `langchain_openai`, `streamlit`, `pydantic`, `python-dotenv`, `pandas`, `altair`.

## Setting Up Environment Variables (.env)
Create a file named `.env` in the root directory of your project (`girl-bot`). This file will contain your API keys and service connection details.

Example `.env` content:

```env
OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
OPENAI_API_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
OPENAI_MODEL_DEPLOYMENT_NAME="YOUR_AZURE_OPENAI_DEPLOYMENT_NAME"
OPENAI_MODEL="OPENAI_MODEL" 
MEM0_API_KEY="YOUR_MEM0_API_KEY"
```

## Running the Streamlit App
After installing dependencies and setting up the `.env` file, run:

```bash
streamlit run app.py
```

This will open the application in your browser, typically at [http://localhost:8501](http://localhost:8501).

## Using the Application
- **Configure Persona:** Use the controls in the sidebar (left) to select "Main Character Traits," "Formality Level," and "Communication Style."

- **Generate Persona:** Click the **"Generate Persona"** button in the sidebar to reset the bot's personality. The conversation will be reset.

- **Chat:** Type your messages in the input field at the bottom of the screen.

- **User Profile:** Talk to the bot about yourself (your name, interests, preferences). Then click **"Show/Update My Profile"** in the sidebar to view your summarized profile generated from Mem0 memories.

- **Suggest Conversation Topics:** Click **"Suggest a Topic"** in the sidebar for the bot to recommend a conversation topic based on your previous preferences.

- **Mood Change History:** Click **"Show Mood History"** in the sidebar to view a graph of your mood fluctuations over time.

---
