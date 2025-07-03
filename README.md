```markdown
# Girls Chatbot Demo

This project presents a Streamlit-based chatbot demonstration application that leverages advanced Mem0 memory features (with a Neo4j graph database and Mem0 Cloud vector database database), and an OpenAI LLM for dynamic persona configuration, mood analysis, and proactive interaction.

## Project Features

- **Dynamic Persona Generation:** Configure the chatbot's personality via the sidebar by selecting core character traits, formality levels, and communication styles.
- **Mood and Intent Analysis:** The bot analyzes your messages to determine your mood and intentions, adapting its responses for a more empathetic interaction.
- **Hybrid Mem0 Memory:** Utilizes both a local Neo4j graph database (running in Docker) for structured relationships like personal connections and events, and Mem0 Cloud vector database (for semantic search and general memory storage).
- **User Profile:** Dynamically generated summary of the user's profile based on stored memories.
- **Proactive Queries:** Two functions for re-engaging in conversation:
  - **"Suggest a Topic"**: General topic suggestions (vector search).
  - **"Generate Proactive Query (Graph Memory)"**: Personalized questions based on recent events and routines from the graph memory.
- **Mood Change History:** Visualization of the user's mood dynamics over time.
- **Content Moderation:** Filtering of controversial topics and potentially undesirable content.

---

## Prerequisites

To run this project, you will need:

- **Docker Desktop** (or Docker Engine and Docker Compose)  
  Ensure Docker is running.

- **Internet connection** (to download Docker images and access OpenAI API and Mem0 Cloud).

- **API Credentials:**
  - **OpenAI API Key:** For accessing OpenAI models (LLM and embeddings).
  - **Mem0 API Key:** For accessing Mem0 Cloud services.
  - **Neo4j Credentials:** You will define these for your local Neo4j instance in the `.env` file (username `neo4j`, and a password you choose).

*(Optional for developers: Python 3.11+ and uv for local development outside Docker.)*

---

## Setup and Running

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/girl-chatbot-demo.git
cd girl-chatbot-demo
```
*(Replace `YOUR_USERNAME/girl-chatbot-demo.git` with the actual URL of your repository.)*

### 2. Configure environment variables

Create a file named `.env` in the root directory of the project (where `docker-compose.yml` is located).  
Copy the content from the `.env.template` file (which you should create in the repository) into your new `.env` file.  
Fill in `.env` with your actual API keys and define your local Neo4j credentials:

```ini
# .env
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
MEM0_API_KEY="YOUR_MEM0_API_KEY"
NEO4J_URI="bolt://neo4j:7687" 
NEO4J_USER="neo4j"
NEO4J_PASSWORD="YOUR_CHOSEN_NEO4J_PASSWORD" Neo4j
NEO4J_DATABASE="neo4j"
```

**ATTENTION:** This `.env` file contains your secrets. **NEVER commit it to Git!**

### 3. Run Docker Containers

In your terminal, navigate to the root directory of the project, then execute:

```bash
docker compose up --build
```

This command will:

- Build the Docker images (if not already built or if there are changes).
- Start all services defined in `docker-compose.yml` (your application, local Neo4j, and Qdrant).

**Note on Neo4j:**  
This command will start a new, local Neo4j database server within a Docker container. You do **not** need to run Neo4j Desktop or any other Neo4j application separately.

### 4. Access the application

After all services have started (this may take a few minutes, especially during the first launch of Neo4j and Qdrant), your Streamlit application will be accessible at:

[http://localhost:8501](http://localhost:8501)

### 5. Stop containers

To stop all running containers and remove them (but preserve database data if you configured volumes in `docker-compose.yml`), execute:

```bash
docker compose down
```

---

## Additional Steps (for developers)

### Local dependency installation

If you plan to actively develop or debug the application outside of Docker, it is recommended to install dependencies locally:

```bash
pip install -e .
```

### Database Inspection

- **Neo4j Browser:**  
  After running `docker compose up`, you can access the Neo4j Browser at [http://localhost:7474](http://localhost:7474).  
  Use the credentials you defined in your `.env` file (`neo4j` as user, your password) to log in.

- **Qdrant UI:**  
  If you are using Qdrant and expose its UI port (usually `6333` or `6334`), you can also access it locally.

---

This updated `README.md` now clearly explains how to set up and run the application with a local Neo4j instance managed by Docker Compose.
```