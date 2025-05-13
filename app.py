import chainlit as cl
from utils.orchestrator import orchestrator_smart
from utils.llm import initialize_llm
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cl.on_chat_start
async def start():
    try:
        llm = initialize_llm()
        cl.user_session.set("llm", llm)
        cl.user_session.set("conversation", [])
        await cl.Message(
            content=("Welcome to your Career Assistant! "
                    "You can:\n"
                    "- Upload your CV\n"
                    "- Ask about jobs\n"
                    "- Get resume help\n"
                    "- Write cover letters")
        ).send()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        await cl.Message(content="Failed to initialize assistant").send()

@cl.on_message  
async def main(message: cl.Message):
    try:
        llm = cl.user_session.get("llm")
        if not llm:
            raise ValueError("LLM not initialized")
            
        msg = cl.Message(content="Processing...")
        await msg.send()

        history = cl.user_session.get("conversation", [])
        combined_prompt = "\n".join(history[-3:]) + f"\nUser: {message.content}"
        # If there's an uploaded file (e.g., CV)
        if message.elements:
            for element in message.elements:
                if element.mime == "application/pdf":
                    file_path = element.path
                    # Forward the file path as part of the message for the orchestrator
                    combined_prompt  += f"\n[CV_PATH]: {file_path}"
        
        response = await orchestrator_smart(combined_prompt, llm)
        msg.content = response or "I didn't get a response. Please try again."
        await msg.update()

        history.append(f"User: {message.content}")
        history.append(f"Assistant: {response}")
        cl.user_session.set("conversation", history)
        
    except Exception as e:
        logger.error(f"Message error: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()