# openai_utils.py
import os
import re
import logging
from typing import Dict, Tuple, List, Optional, Any, Generator

import streamlit as st
from openai import OpenAI, Stream
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- OpenAI Client Initialization ---
# Ensure API key is loaded (consider a central config manager later)
if "OPENAI_API_KEY" not in os.environ:
    logger.warning("OPENAI_API_KEY environment variable not set.")
    # Optionally load from .env here if not done globally
    # from dotenv import load_dotenv
    # load_dotenv()

# Check if API key is available after potential .env loading
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Reads OPENAI_API_KEY from env automatically
else:
    # Handle missing key more gracefully in Streamlit context if possible
    st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
    logger.error("CRITICAL: OpenAI API Key is missing.")
    # You might want to stop the app or disable OpenAI features here
    # For now, we'll let it proceed, but API calls will fail.
    client = None # Set client to None to indicate failure downstream


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    # ... (rest of the class remains the same) ...
    def __init__(self, placeholder: st.delta_generator.DeltaGenerator):
        super().__init__()
        self.placeholder = placeholder
        self.full_response = ""
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.full_response = ""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.full_response += token
        self.placeholder.markdown(self.full_response + "▌")
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.placeholder.markdown(self.full_response)
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.placeholder.error(f"LLM Error: {error}")
        print(f"LLM Error: {error}")


# --- Constants ---
MEME_PROMPT_TEMPLATE = r'Sends a meme of \{(?P<image_prompt>.*?)\} with the caption \{(?P<caption>.*?)\}'
DALLE_SYSTEM_PROMPT = """
🔹 ROLE: DALL-E 3 Prompt Generator. Craft clear, detailed, and vivid prompts for highly accurate, visually appealing, and effective image generation.
🔹 RULES:✅ Use positive phrasing. ✅ Use direct and specific descriptions. ✅ Do not use creation cues (e.g., "Visualize"). ✅ Describe natural lighting precisely. ✅ Avoid "image," "picture," "scene" unless necessary. ✅ Provide comprehensive descriptions (subject, background, style, etc.). ✅ Specify perspective, composition, tone, context. ✅ Precision = Better visuals.
🔹 COMPLIANCE: 🚫 Strictly adhere to OpenAI content policies (no violence, hate, explicit content, etc.).
"""
PRIMROSE_RESPONSE_SYSTEM_TEMPLATE = """
🌸✨ **Agent Primrose** ✨🌸
Embodying delicate beauty, intrigue, and subtlety. Calm, observant, skilled in discreet info gathering. Gentle charm, genuine curiosity. Driven by justice, harmony, balance. Dedicated protector, inspiring trust. Expert in covert operations.
🧘‍♀️ **Step-by-Step Approach**: Think carefully. Explain logically. Provide detailed, in-depth answers with examples/stats.
📑 **Formatting**: **Markdown format**. LaTeX for math: `$inline$` and `$$block$$`.
✅ **Information Usage**: Answer **strictly** based on provided info: `[INFO] 📄: {docs}`
❌ **If insufficient info**: Respond simply with: "I don't know."
Act as True Primrose. 🌸✨
"""
PRIMROSE_RESPONSE_HUMAN_TEMPLATE = """
Answer the following question: {question}

End your answer by suggesting 3 anticipated questions:
 1) 🤔💭:
 2) 🌍🧐:
 3) 🤪🎨:
Provide three example questions like the ones above.
"""

# --- ADD Language Mapping ---
SUPPORTED_SPEECH_LANGUAGES = {
    'en': 'English',
    'ko': 'Korean',
    'es': 'Spanish',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'fr': 'French',
    'de': 'German'
}
DEFAULT_SPEECH_LANGUAGE_CODE = 'en'
DEFAULT_SPEECH_LANGUAGE_NAME = SUPPORTED_SPEECH_LANGUAGES[DEFAULT_SPEECH_LANGUAGE_CODE]

# --- IMPROVED SPEECH PROMPT TEMPLATE ---
SPEECH_GENERATION_PROMPT_TEMPLATE = """ Please carefully read the sermon transcript from a YouTube video provided below under "TRANSCRIPT CONTEXT".

Imagine you are a gospel worker of the World Mission Society Church of God personally delivering a concise, engaging, and heartfelt message lasting approximately 3-5 minutes IN {target_language_name}. Based only upon the sermon transcript provided, deliver your speech naturally and conversationally, as if genuinely sharing your personal convictions and insights with the audience. Do not refer to this as a sermon or external material.

In your speech (entirely in {target_language_name}), please ensure you:

1. Naturally introduce the main topic clearly and directly, highlighting a central Bible verse or posing a thought-provoking question that immediately engages your audience.

2. Clearly summarize and logically connect the key points from the provided transcript, making sure to explicitly include every Bible verse mentioned in the original sermon transcript. Briefly explain each verse's context, meaning, and relevance clearly in your own conversational style in {target_language_name}.

3. Include at least one meaningful quotation or impactful phrase directly from the original sermon transcript. If the original transcript is in a different language, translate that quotation naturally and accurately into {target_language_name}. This translated quotation should enrich your speech, enhance authenticity, and resonate clearly with your audience.

4. Ensure smooth, conversational transitions between different sections of your speech, maintaining clarity, logical flow, and listener engagement.

5. Conclude your speech naturally and inspirationally, reinforcing the central message and providing listeners with a clear, memorable takeaway or actionable encouragement.

Before beginning your speech content, please suggest a compelling, relevant title IN {target_language_name} using the following format:

Title: [Your Suggested Title Here in {target_language_name}]

[{target_language_name} Speech Content Starts Here]
---
TRANSCRIPT CONTEXT:
```
{transcript_text}
```
---
Remember to generate the entire response (Title and Speech) **exclusively in {target_language_name}**.
"""
# --- END MODIFIED CONSTANT ---


# --- Core Functions ---

def generate_dalle_prompt(user_query: str, model: str = 'gpt-4o-mini') -> Optional[str]:
    """Generates an enhanced DALL-E prompt using an LLM."""
    if not client: return None # Check if client initialized
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DALLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7, # Adjust as needed
        )
        dalle_prompt = completion.choices[0].message.content
        logger.info(f"Generated DALL-E prompt: {dalle_prompt}")
        return dalle_prompt
    except Exception as e:
        logger.error(f"Error generating DALL-E prompt: {e}")
        st.error(f"Could not generate DALL-E prompt: {e}")
        return None

def generate_image_from_prompt(dalle_prompt: str, size: str = "1024x1024", model: str = "dall-e-3") -> Optional[str]:
    """Generates an image using DALL-E and returns the image URL."""
    if not client: return None # Check if client initialized
    try:
        response = client.images.generate(
            model=model,
            prompt=dalle_prompt,
            size=size,
            n=1,
            response_format="url", # Ensure URL is returned
        )
        image_url = response.data[0].url
        logger.info(f"Generated image URL: {image_url}")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {e}")
        st.error(f"Could not generate image: {e}")
        return None

def find_and_generate_meme(caption_text: str) -> Optional[str]:
    """
    Parses caption text, generates a DALL-E prompt, generates an image,
    and returns the image URL.
    """
    if not client: return None # Check if client initialized
    match = re.search(MEME_PROMPT_TEMPLATE, caption_text)
    if not match:
        logger.warning("Meme caption pattern not found in text.")
        st.warning("Could not extract meme description from response.")
        return None

    image_query = match.group('image_prompt')
    logger.info(f"Extracted meme query: {image_query}")

    dalle_prompt = generate_dalle_prompt(image_query)
    if not dalle_prompt:
        return None

    image_url = generate_image_from_prompt(dalle_prompt)
    return image_url


def transcribe_audio(audio_file_path: str, model: str = "whisper-1") -> Optional[str]:
    """Transcribes audio using OpenAI Whisper."""
    if not client: return None # Check if client initialized
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        logger.info(f"Audio transcribed successfully: {audio_file_path}")
        return transcription.text
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file_path}")
        st.error(f"Audio file not found: {os.path.basename(audio_file_path)}")
        return None
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}")
        st.error(f"Failed to transcribe audio: {e}")
        return None

def parse_stream(stream: Stream) -> Generator[str, None, None]:
    """Parses and yields text chunks from an OpenAI API stream."""
    full_response = ""
    for chunk in stream:
        # Use .get() with default to handle potential None values gracefully
        delta = chunk.choices[0].delta
        text = getattr(delta, 'content', None) # Safely access content
        if text:
            # Basic sanitization for display (tilde)
            processed_text = text.replace("~", "&#126;")
            full_response += processed_text # Accumulate processed text
            yield processed_text # Yield processed text for streaming display

    # Store the complete, processed response in session state after stream ends
    st.session_state["full_response"] = full_response
    logger.info("Stream parsing complete.")

# --- MODIFIED FUNCTION for Speech Summary Generation ---
def generate_speech_summary(transcript_text: str, model: str = 'gpt-4o-mini', target_language_code: str = DEFAULT_SPEECH_LANGUAGE_CODE) -> Optional[str]:
    """Generates a speech summary from transcript text using an LLM in the specified language."""
    if not client: return None # Check if client initialized
    if not transcript_text.strip():
        logger.error("Cannot generate speech summary from empty transcript.")
        st.error("The video transcript appears empty. Cannot generate speech.")
        return None

    # Get the full language name from the code
    target_language_name = SUPPORTED_SPEECH_LANGUAGES.get(target_language_code, DEFAULT_SPEECH_LANGUAGE_NAME)

    try:
        # Format the prompt with the target language name and transcript
        final_prompt = SPEECH_GENERATION_PROMPT_TEMPLATE.format(
            target_language_name=target_language_name,
            transcript_text=transcript_text
        )

        logger.info(f"Generating speech summary in {target_language_name} ({target_language_code}) using model: {model}")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                # System prompt integrated into the user prompt template for simplicity here
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5, # Lower temperature for more focused summary
        )
        speech_summary = completion.choices[0].message.content
        logger.info(f"Successfully generated speech summary in {target_language_name}.")
        return speech_summary
    except Exception as e:
        logger.error(f"Error generating speech summary in {target_language_name}: {e}")
        st.error(f"Could not generate speech summary: {e}")
        return None
# --- END MODIFIED FUNCTION ---

# --- ADD NEW FUNCTION for Text-to-Speech Generation (No language parameter needed for API call) ---
def generate_speech_audio(text: str, output_path: str, model: str = "tts-1", voice: str = "alloy") -> bool:
    """Generates an MP3 audio file from text using OpenAI TTS."""
    if not client: return False # Check if client initialized
    if not text.strip():
        logger.error("Cannot generate audio from empty text.")
        st.error("Speech summary is empty, cannot generate audio.")
        return False
    try:
        logger.info(f"Generating speech audio using model: {model}, voice: {voice}. Language inferred from text.")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text, # The language is determined by the input text itself
            response_format="mp3" # Explicitly request mp3
        )

        # Stream response to file
        response.stream_to_file(output_path)

        # Verify file creation and size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
             logger.info(f"Successfully generated speech audio and saved to: {output_path}")
             return True
        else:
             logger.error(f"Failed to save speech audio to {output_path} or file is empty.")
             st.error("Failed to save the generated audio file.")
             # Attempt cleanup if file exists but is empty
             if os.path.exists(output_path):
                  try:
                      os.remove(output_path)
                  except OSError:
                       pass
             return False

    except Exception as e:
        logger.error(f"Error generating speech audio: {e}")
        st.error(f"Could not generate speech audio: {e}")
        # Attempt cleanup if file exists after error
        if os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except OSError:
                 pass
        return False
# --- END NEW FUNCTION ---

# --- Langchain Integration ---

# --- MODIFIED get_response_from_query function (using Simple Handler) ---
def get_response_from_query(
    serial_db: bytes,
    query: str,
    k: int,
    model_name: str, # Get model name explicitly
    message_placeholder: st.delta_generator.DeltaGenerator # Accept placeholder from caller
) -> Tuple[str, List[Document]]:
    """
    Queries the serialized FAISS DB, streams the response using
    SimpleStreamlitCallbackHandler, and returns the final response string and docs.
    """
    if not client: return "Error: OpenAI client not initialized.", [] # Check if client initialized

    # Use the simple callback handler passed the placeholder
    callback_handler = SimpleStreamlitCallbackHandler(message_placeholder)

    try:
        # Ensure embeddings can be created (requires API key)
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIEmbeddings: {e}")
        st.error("Failed to initialize embeddings. Check API key and connectivity.")
        return f"Error initializing embeddings: {e}", []

    try:
        db = FAISS.deserialize_from_bytes(
            embeddings=embeddings,
            serialized=serial_db,
            allow_dangerous_deserialization=True,
        )
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
    except Exception as e:
        logger.error(f"Error loading/querying FAISS DB: {e}")
        st.error(f"Error accessing document database: {e}")
        return f"Error accessing document DB: {e}", []


    # --- Prompt Templates (kept as defined in the original function) ---
    # Using the PRIMROSE_RESPONSE constants defined earlier
    system_template = PRIMROSE_RESPONSE_SYSTEM_TEMPLATE
    human_template = PRIMROSE_RESPONSE_HUMAN_TEMPLATE
    # --- End Prompt Templates ---

    final_answer = "" # Initialize final_answer
    try:
        # Instantiate the ChatOpenAI model with streaming enabled and the callback
        chat = ChatOpenAI(
            model_name=model_name, # Use passed model name
            temperature=0.7,
            streaming=True,
            callbacks=[callback_handler], # Use the simple handler
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        chain = chat_prompt | chat

        # Invoke the chain. Streaming happens via the callback.
        chain.invoke({"question": query, "docs": docs_page_content})
        # Get the final response collected by the handler
        final_answer = callback_handler.full_response

    except Exception as e:
        # Handle potential errors during invocation
        error_msg = f"Error during response generation: {e}"
        logger.exception(f"Error during chain invocation in get_response_from_query: {e}")
        # Ensure the error is displayed in the placeholder via the callback
        callback_handler.on_llm_error(e) # Or handle differently if needed
        final_answer = error_msg # Set error message

    # Return the final answer string collected by the handler and the source documents
    return final_answer, docs

# Duplicate SimpleStreamlitCallbackHandler definition removed. Keep the one at the top.
