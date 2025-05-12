# JoBot
JoBot – AI Career Advisor built as part of the AI for Education project at ESPRIT School of Engineering. This assistant helps students explore careers, develop skills, and plan their professional paths using intelligent recommendations.
## AI Career Assistant for ESPRIT Students JoBot
# Overview
This project is an AI-powered career assistant chatbot built to support ESPRIT students in navigating their professional journeys. It provides intelligent guidance in CV review, job search, cover letter writing, and more. Leveraging a multi-agent system with an orchestrator, the assistant ensures personalized, context-aware interactions.

It’s built with LangChain, Chainlit, and optionally integrates Groq for efficient inference, providing fast, cost-effective alternatives to proprietary APIs like OpenAI .

# Features
🧠 Multi-agent system: Specialized agents handle tasks like CV analysis, job suggestions, cover letter generation, etc.

🧭 Smart routing: An LLM-powered orchestrator dynamically routes user messages to the right agent.

📄 CV upload and analysis: Users can upload their PDF CVs for instant review and improvement suggestions.

💼 Job guidance: Personalized job search tips, recommendations, and career advice.

✍️ Cover letter generator: Generates personalized cover letters based on user input and job roles.

💬 Context memory: Short-term memory to track the last K turns for context-aware responses.

🖥️ Chainlit UI: Real-time conversational interface with support for file uploads and response streaming.

# Architecture

-Chat Start:
Initializes the base LLM (Groq or vLLM via OpenAI-compatible API).
Sends a welcome message and stores the LLM in session.

-Message Handling:
Uses a central function orchestrator_smart to classify and route the message.
CVs (PDFs) are handled via message.elements, and paths are forwarded to the orchestrator.

-Routing:
The orchestrator LLM decides which agent to invoke (e.g., CVAgent, CoverLetterAgent, JobAgent).
Each agent is re-instantiated with its own memory buffer (K=3) for context.

-Memory:
Agents use ConversationBufferMemory to maintain limited context per user.
Additionally, a global cl.user_session memory (conversation) tracks the last few exchanges across sessions.

-Response Generation:
Responses are streamed to the user.
The chat history is optionally stored and shown on resume.

# How to Run
Install dependencies:
pip install -r requirements.txt

Start Chainlit:
chainlit run app.py

# File Structure
project/
├── agents/
│   ├── __init__.py
│   ├── information.py
│   ├── motivation.py
│   ├── cv_enhancer.py
│   └── job_matching.py
├── tools/
│   ├── __init__.py
│   ├── documents.py
│   ├── search.py
│   ├── web.py
├── utils/
│   ├── __init__.py
│   ├── llm.py
│   ├── orchestrator.py
│   └── config.py
├── data/
│   ├── __init__.py
│   ├── docs.py
│   ├── file1.json
│   └── file2.json
│   └── file3.json
│   └── file4.json
.
.
├── app.py

# Example Prompt Flow
User: Can you check my CV?
↓
Orchestrator: Routes to CVAgent
↓
CVAgent: Analyzes uploaded CV and provides suggestions

Credits
Developed by the AI student team Neural Leaders at ESPRIT, as part of an initiative to modernize and personalize career guidance for students using AI.
