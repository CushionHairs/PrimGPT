# --- START OF REFACTORED FILE PrimGPT.py ---

import os
import logging
from typing import Tuple, Dict, Any, List

# External Libraries
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import requests
from io import BytesIO
import dill as pickle # For loading document DBs
from urllib.parse import quote_plus
import webbrowser # Required for opening tab
import time # For potential delays

# Local Modules
from chat_config import get_template_config # Centralized config loading
from utils import ( # General utilities
    set_bg_hack,
    sidebar_bg,
    image_to_base64,
    display_meme,
    ani_newLink,
    validate_youtube_url,
    safe_filename,
)
from openai_utils import ( # OpenAI related functions
    parse_stream,
    find_and_generate_meme,
    get_response_from_query,
    client, # Re-export OpenAI client if needed directly
    generate_speech_summary,
    generate_speech_audio,
    # --- ADD NEW IMPORTS ---
    SUPPORTED_SPEECH_LANGUAGES, # Import the language map
    DEFAULT_SPEECH_LANGUAGE_CODE, # Import default lang code
    # --- END NEW IMPORTS ---
)
from youtube_utils import ( # YouTube functions
    yt_search,
    yt_ready, # Assumed to return True on success, False on failure and handle its own st.error/warning
    generate_y2mate_link,
    yt_download,
    YOUTUBE_DB_PATH, # Constants for paths
    YOUTUBE_DB_FILENAME,
    YOUTUBE_DOWNLOAD_PATH, # Ensure this is imported/used consistently
    get_youtube_video_title,
    load_transcript # Need this directly for the speech function
)
from document_utils import ( # Document functions
    file_processing_pipeline,
    DOC_DATA_PATH, # Constants for paths
)
from web_utils import fetch_main_content # Web scraping
from web_utils import fetch_main_content_bs4 
from web_utils import fetch_main_content_selenium
# --- Configuration & Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure data directories exist
os.makedirs(DOC_DATA_PATH, exist_ok=True)
os.makedirs(YOUTUBE_DB_PATH, exist_ok=True)
os.makedirs(YOUTUBE_DOWNLOAD_PATH, exist_ok=True) # Also ensure YT download path exists

# Define constants for session state keys
STATE_OPENAI_MODEL = "openai_model"
STATE_TOKEN_SIZE = "token_size"
STATE_CHAT_MESSAGES = "messages"
STATE_FULL_RESPONSE = "full_response"
STATE_MEME_CAPTION = "caption"
STATE_DOC_MESSAGES = "doc_messages"
STATE_SELECTED_DOC_PATH = "selected_doc_path"
STATE_YT_MESSAGES = "yt_messages"
STATE_YT_DB_READY = "yt_db_ready" # Track if YT DB exists and processing succeeded
STATE_YT_CHAT_URL_INPUT = 'yt_chat_url_input' # Define key for clarity
# --- ADD NEW STATE KEYS for Speech Download Persistence ---
STATE_SPEECH_READY = 'speech_ready'
STATE_SPEECH_LANG_CODE = 'speech_lang_code' # Store selected language code
STATE_SPEECH_TXT_PATH = 'speech_txt_path'
STATE_SPEECH_MP3_PATH = 'speech_mp3_path'
STATE_SPEECH_USER_TXT_FILENAME = 'speech_user_txt_filename'
STATE_SPEECH_USER_MP3_FILENAME = 'speech_user_mp3_filename'
# --- END NEW STATE KEYS ---

# --- Helper Functions ---

# --- REVISED initialize_session_state ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    # yt_db_physical_exists = os.path.exists(os.path.join(YOUTUBE_DB_PATH, YOUTUBE_DB_FILENAME)) # Not needed for init logic
    defaults = {
        STATE_OPENAI_MODEL: "gpt-4o-mini",
        STATE_TOKEN_SIZE: 128, # Example, adjust based on model choice logic
        STATE_CHAT_MESSAGES: [],
        STATE_FULL_RESPONSE: "",
        STATE_MEME_CAPTION: "",
        STATE_DOC_MESSAGES: [], # Initialize empty, add system prompt later if needed
        STATE_SELECTED_DOC_PATH: None,
        STATE_YT_MESSAGES: [], # Initialize empty, add system prompt later if needed
        STATE_YT_DB_READY: False,
        STATE_YT_CHAT_URL_INPUT: '',
        # --- Initialize new speech state keys ---
        STATE_SPEECH_READY: False,
        STATE_SPEECH_LANG_CODE: DEFAULT_SPEECH_LANGUAGE_CODE, # Default language
        STATE_SPEECH_TXT_PATH: None,
        STATE_SPEECH_MP3_PATH: None,
        STATE_SPEECH_USER_TXT_FILENAME: None,
        STATE_SPEECH_USER_MP3_FILENAME: None,
        # --- End initialization ---
        'app_initialized': False # Ensure this is handled correctly elsewhere if used
    }
    # Initialize only if the app hasn't been initialized in this session
    if not st.session_state.get('app_initialized'):
        logger.info("Initializing session state for the first time.")
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        st.session_state['app_initialized'] = True # Mark as initialized
    else:
         # Ensure keys exist even if added later (e.g., during development)
         for key, value in defaults.items():
             if key not in st.session_state:
                  logger.warning(f"Session state key '{key}' missing, initializing.")
                  st.session_state[key] = value

    # Ensure system prompts are added if message lists are empty after initialization
    if STATE_CHAT_MESSAGES in st.session_state and not st.session_state[STATE_CHAT_MESSAGES]:
         # Reset chat will handle adding the correct system prompt based on template
         pass # Reset logic handles this
    if STATE_DOC_MESSAGES in st.session_state and not st.session_state[STATE_DOC_MESSAGES]:
         st.session_state[STATE_DOC_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]
    if STATE_YT_MESSAGES in st.session_state and not st.session_state[STATE_YT_MESSAGES]:
         st.session_state[STATE_YT_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]


def setup_page_config():
    """Sets Streamlit page configuration and background styles."""
    st.set_page_config(layout="wide", page_title="PrimGPT", page_icon="🌸")
    try:
        # Ensure paths are correct relative to PrimGPT.py
        assets_dir = "assets"
        if not os.path.exists(os.path.join(assets_dir, "background.jpg")):
             raise FileNotFoundError("Background image not found.")
        set_bg_hack(os.path.join(assets_dir, "background.jpg"))
        if not os.path.exists(os.path.join(assets_dir, "sidebar_background.jpg")):
             raise FileNotFoundError("Sidebar background image not found.")
        sidebar_bg(os.path.join(assets_dir, "sidebar_background.jpg"))
    except FileNotFoundError as e:
        logger.warning(f"{e}. Using default background.")
        st.warning("Background images not found in 'assets' folder. Using default style.")
    except Exception as e:
        logger.error(f"Error setting background: {e}")
        st.warning(f"Could not set custom backgrounds: {e}")


def render_sidebar() -> Tuple[str, str]:
    """Renders the sidebar components and returns selected AI model and function."""
    with st.sidebar:
        try:
            assets_dir = "assets"
            # Display decorative images
            st.image(os.path.join(assets_dir, "title.png"), width=220)
            st.image(os.path.join(assets_dir, "primrose.png"), caption="Agent Primrose", width=220)
        except FileNotFoundError:
             logger.warning("Sidebar image files not found in 'assets'.")
             st.warning("Sidebar images not found.")
        except Exception as e:
             logger.error(f"Error loading sidebar images: {e}")
             st.warning("Could not load sidebar images.")

        # Model selection - Updated model names
        ai_model_options = {
            'gpt-4.1': 'gpt-4.1(Best)',
            'gpt-4.1-mini': 'gpt-4.1(Middle)',
            'gpt-4.1-nano': 'gpt-4.1(Light)',
            'gpt-4.5-preview': 'gpt-4.5(Best)',
            'o3-mini': 'o3-mini(Logical)',
            'o4-mini': 'o4-mini(Logical)',
            # Add other models if needed
        }
        selected_model_key = st.radio(
            "Select AI engine 👇",
            options=list(ai_model_options.keys()),
            format_func=lambda key: ai_model_options[key],
            horizontal=True,
            # label_visibility="collapsed", # Keep label for clarity
            key="ai_engine_selector" # Add key for robustness
        )
        st.session_state[STATE_OPENAI_MODEL] = selected_model_key

        # Update token size based on selection (example values, adjust as needed)
        # Note: Token size might refer to context window, not always output limit. Clarify usage.
        if selected_model_key == 'gpt-4.1':
            st.session_state[STATE_TOKEN_SIZE] = 30000 # Context window size
        elif selected_model_key == 'gpt-4.1-mini':
            st.session_state[STATE_TOKEN_SIZE] = 200000 # Context window size
        elif selected_model_key == 'gpt-4.1-nano':
            st.session_state[STATE_TOKEN_SIZE] = 200000 # Context window size
        elif selected_model_key == 'gpt-4.5-preview':
            st.session_state[STATE_TOKEN_SIZE] = 128000 # Context window size
        elif selected_model_key == 'o3-mini':
            st.session_state[STATE_TOKEN_SIZE] = 200000 # Context window size
        elif selected_model_key == 'o4-mini':
            st.session_state[STATE_TOKEN_SIZE] = 200000 # Context window size
        else:
             st.session_state[STATE_TOKEN_SIZE] = 4096 # Default fallback


        # Function selection
        # Determine available functions based on potential model capabilities (example)
        if st.session_state[STATE_OPENAI_MODEL] in ['gpt-4.5-preview', 'gpt-4.1', 'gpt-4.1 (long context)', 'gpt-4.1-mini', 'gpt-4.1-mini (long context)', 'gpt-4.1-nano', 'gpt-4.1-nano (long context)']: 
             function_options = ['Chat', 'Youtube', 'Document']
        else:
             function_options = ['Chat'] # Fallback

        selected_function = st.radio(
            "Select function 👇",
            function_options,
            horizontal=True,
            # label_visibility="collapsed", # Keep label
            key="function_selector"
        )

        # Render function-specific sidebar items
        if selected_function == 'Chat':
            render_chat_sidebar()
        elif selected_function == 'Youtube':
            render_youtube_sidebar()
        elif selected_function == 'Document':
            render_document_sidebar()

        st.divider()
        render_wolfram_widget()

        return st.session_state[STATE_OPENAI_MODEL], selected_function

def render_chat_sidebar():
    """Renders sidebar elements specific to the Chat function."""

    # Template Selection
    template_names = list(get_template_config("Agent Primrose")) # Get list of names from config keys
    # Ensure 'Agent Primrose' is first if it exists, or handle default
    # For now, assumes TEMPLATE_CONFIGS keys are used directly
    from chat_config import TEMPLATE_CONFIGS # Import the dict directly
    template_names = list(TEMPLATE_CONFIGS.keys())

    template_name = st.selectbox(
        "Choose a Chat Persona:",
        template_names,
        index=template_names.index(st.session_state.get('selected_template', "Agent Primrose")), # Default selection
        key="template_selector",
        # label_visibility="collapsed"
    )
    # Update selected template immediately if changed
    if st.session_state.get('selected_template') != template_name:
         st.session_state['selected_template'] = template_name
         # Optionally reset chat when persona changes? Or keep history? For now, keep history.
         # reset_chat(template_name)
         # st.toast(f"Switched persona to {template_name}", icon="🎭")
         # st.rerun() # Rerun might be too disruptive here, let reset button handle clearing

    # Reset Chat Button
    if st.button("✨ New Chat", key="new_chat_button"):
        selected_template = st.session_state.get('selected_template', "Agent Primrose")
        reset_chat(selected_template) # Pass current template name
        st.toast("Chat reset!", icon="🔄")
        st.rerun() # Rerun needed to clear main panel

    st.divider()

    # Special controls for specific templates
    if st.session_state.get('selected_template') == "Anime Finder":
        query = st.text_input("Enter Animation title:", key="anime_query")
        if st.button("Search Anime", key="anime_search_button"):
            if query:
                ani_newLink(query) # Assumes ani_newLink handles display
            else:
                st.warning("Please enter an animation title to search.")

    # Image Input (Check vision capabilities - gpt-4o and gpt-4o-mini are vision models)
    if st.session_state[STATE_OPENAI_MODEL] in ['gpt-4.5-preview']:
        st.caption("🖼️ Add an Image (Optional)")
        option = st.radio(
            "Image Source:",
            ["Upload Image", "Image URL"],
            horizontal=True,
            label_visibility="collapsed",
            key="image_source_option"
            )

        uploaded_file = None
        image_url = None
        image_to_process = None # Local variable for this render cycle

        if option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "webp", "gif"],
                label_visibility="collapsed",
                key="image_uploader"
                )
            if uploaded_file:
                try:
                    image_to_process = Image.open(uploaded_file)
                    st.image(image_to_process, caption='Preview Uploaded Image', use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading uploaded image: {e}")
                    image_to_process = None # Reset on error

        elif option == "Image URL":
            image_url = st.text_input("Enter Image URL:", key="image_url_input")
            # Trigger URL loading only if URL is provided and button is clicked
            if image_url and st.button("Load Image from URL", key="load_image_url_button"):
                try:
                    # Add User-Agent header
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    response = requests.get(image_url, headers=headers, timeout=10, stream=True)
                    response.raise_for_status()
                    # Check content type before loading
                    content_type = response.headers.get('content-type')
                    if content_type and 'image' in content_type.lower():
                         image_to_process = Image.open(BytesIO(response.content))
                         st.image(image_to_process, caption='Preview Image from URL', use_container_width=True)
                    else:
                         st.error(f"Invalid content type received: {content_type}. URL does not point to a valid image.")
                         image_to_process = None
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to retrieve image: {e}")
                    image_to_process = None
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")
                    image_to_process = None
            elif not image_url and st.session_state.get("load_image_url_button"): # If button clicked without URL
                 st.warning("Please enter an image URL.")


        # Store the loaded image in session state if it's valid in this cycle
        # This ensures the image is available when the chat input is processed
        if image_to_process:
             st.session_state['image_to_process'] = image_to_process
        # If no valid image was loaded in this cycle, ensure the state is cleared
        # Check if upload/URL process finished WITHOUT a valid image
        elif uploaded_file is None and not (image_url and image_to_process): # More specific condition
             if 'image_to_process' in st.session_state:
                  del st.session_state['image_to_process']


# --- REVISED render_youtube_sidebar ---
def render_youtube_sidebar():
    """Renders sidebar elements specific to the YouTube function."""

    # --- YouTube Search ---
    st.caption("🔍 Find YouTube Videos")
    yt_find = st.text_input(
        "Search YouTube Videos",
        placeholder='Enter search query...',
        label_visibility='collapsed',
        key='yt_search_query'
    )
    if st.button("Search Videos", key='yt_search_button'):
        # Clear previous results and state before new search
        if 'yt_search_results' in st.session_state:
            del st.session_state['yt_search_results']
        st.session_state[STATE_YT_DB_READY] = False # Reset analysis readiness
        # Reset speech state
        st.session_state[STATE_SPEECH_READY] = False
        st.session_state[STATE_SPEECH_TXT_PATH] = None
        st.session_state[STATE_SPEECH_MP3_PATH] = None
        # Clear any previous status messages related to speech/analysis
        # Find relevant placeholders if they exist, otherwise ignore
        # status_placeholder_speech.empty() # Requires placeholder defined earlier

        yt_search(yt_find) # Execute search
        st.rerun() # Rerun to show search results / clear previous UI state

    st.divider()

    # --- YouTube Chat / Analysis / Speech ---
    st.caption("📊 Analyze or Summarize Video")
    st.session_state[STATE_YT_CHAT_URL_INPUT] = st.text_input(
        "Analyze or Summarize YouTube Video",
        placeholder='Paste video URL here...',
        label_visibility='collapsed',
        key='yt_chat_url_display',
        value=st.session_state.get(STATE_YT_CHAT_URL_INPUT, '')
    )
    yt_chat_url_from_input = st.session_state[STATE_YT_CHAT_URL_INPUT]

    # --- ADD Language selection dropdown ---
    selected_lang_code = st.selectbox(
        "Select Speech Language:",
        options=list(SUPPORTED_SPEECH_LANGUAGES.keys()),
        format_func=lambda code: SUPPORTED_SPEECH_LANGUAGES[code], # Show full names
        key='speech_language_select',
        index=list(SUPPORTED_SPEECH_LANGUAGES.keys()).index(st.session_state.get(STATE_SPEECH_LANG_CODE, DEFAULT_SPEECH_LANGUAGE_CODE)) # Pre-select stored value
    )
    # Update state immediately if selection changes
    if st.session_state.get(STATE_SPEECH_LANG_CODE) != selected_lang_code:
        st.session_state[STATE_SPEECH_LANG_CODE] = selected_lang_code
        logger.info(f"Speech language selection changed to: {selected_lang_code}")
        # Reset speech readiness if language changes? Maybe not necessary.
        # st.session_state[STATE_SPEECH_READY] = False
        # st.rerun() # Avoid rerun just for selection change

    # --- Placeholder for Status Messages (speech generation) ---
    status_placeholder_speech = st.empty()

    # --- Action Buttons ---
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔬 Prepare", key='yt_ready_button', help="Analyze transcript for Q&A"):
            current_url = st.session_state.get(STATE_YT_CHAT_URL_INPUT, '')
            validated_url = validate_youtube_url(current_url)
            if validated_url:
                # Reset states before preparation
                st.session_state[STATE_SPEECH_READY] = False
                st.session_state[STATE_SPEECH_TXT_PATH] = None
                st.session_state[STATE_SPEECH_MP3_PATH] = None
                st.session_state[STATE_YT_DB_READY] = False # Explicitly set to false before attempt
                st.session_state[STATE_YT_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}] # Reset chat history
                status_placeholder_speech.empty() # Clear any previous speech status

                analysis_success = False # Flag
                yt_db_file = os.path.join(YOUTUBE_DB_PATH, YOUTUBE_DB_FILENAME)

                with st.spinner(f"Preparing analysis for {current_url}..."):
                    try:
                        # yt_ready handles internal errors/logging and returns bool/None
                        analysis_success = yt_ready(current_url)
                        # Optional: Add explicit file check after call returns for robustness
                        if analysis_success and not os.path.exists(yt_db_file):
                             logger.warning(f"yt_ready returned success, but DB file {yt_db_file} not found.")
                             analysis_success = False # Override success if file missing
                             st.error("Analysis preparation step completed, but the data file is missing.")

                    except Exception as e:
                        logger.error(f"Exception during yt_ready call for {current_url}: {e}", exc_info=True)
                        st.error(f"An unexpected error occurred during video preparation: {e}")
                        analysis_success = False

                st.session_state[STATE_YT_DB_READY] = analysis_success # Update state based on outcome
                if analysis_success:
                    st.toast("Video analysis prepared!", icon="✅")
                else:
                    # yt_ready or the logic above should have shown specific error
                    st.warning("Failed to prepare video for analysis. Check URL/transcript/logs.")
                st.rerun() # Rerun to reflect state changes (e.g., enable Q&A)

    with col2:
        # --- REVISED "A brief speech" Button Logic ---
        if st.button("🗣️ Speech", key='yt_brief_speech_button', help="Generate a speech summary and audio"):
            current_url = st.session_state.get(STATE_YT_CHAT_URL_INPUT, '')
            validated_url = validate_youtube_url(current_url)

            if validated_url:
                # Retrieve selected language
                target_lang_code = st.session_state.get(STATE_SPEECH_LANG_CODE, DEFAULT_SPEECH_LANGUAGE_CODE)
                target_lang_name = SUPPORTED_SPEECH_LANGUAGES.get(target_lang_code, DEFAULT_SPEECH_LANGUAGE_CODE)
                logger.info(f"Starting speech generation for {validated_url} in {target_lang_name} ({target_lang_code})")

                # Reset state before starting
                st.session_state[STATE_SPEECH_READY] = False
                st.session_state[STATE_SPEECH_TXT_PATH] = None
                st.session_state[STATE_SPEECH_MP3_PATH] = None
                status_placeholder_speech.info(f"Generating {target_lang_name} speech...")

                success_flag = False # Track overall success
                video_title = None
                fixed_txt_path = None
                fixed_mp3_path = None
                user_facing_txt_filename = f"speech_summary_{target_lang_code}.txt"
                user_facing_mp3_filename = f"speech_summary_{target_lang_code}.mp3"

                with st.spinner(f"Processing for {target_lang_name} speech..."):
                    try:
                        # 1. Get Video Title (for filename)
                        video_title = get_youtube_video_title(validated_url)
                        base_filename = safe_filename(video_title if video_title else "youtube_video")
                        user_facing_txt_filename = f"{base_filename}_speech_{target_lang_code}.txt"
                        user_facing_mp3_filename = f"{base_filename}_speech_{target_lang_code}.mp3"

                        # Use language code in fixed internal filenames to avoid overwrites between languages
                        fixed_base_filename = f"speech_summary_{target_lang_code}"
                        fixed_txt_path = os.path.join(YOUTUBE_DOWNLOAD_PATH, f"{fixed_base_filename}.txt")
                        fixed_mp3_path = os.path.join(YOUTUBE_DOWNLOAD_PATH, f"{fixed_base_filename}.mp3")

                        # 2. Load Transcript (without forcing translation)
                        status_placeholder_speech.info("Loading transcript...")
                        transcript_data = load_transcript(validated_url, translation=None) # Crucial: translation=None

                        if transcript_data:
                            transcript_string = " ".join(entry.get("text", "") for entry in transcript_data)

                            if transcript_string.strip():
                                # 3. Generate Speech Summary in Target Language
                                status_placeholder_speech.info(f"Generating {target_lang_name} summary (LLM)...")
                                speech_text_generated = generate_speech_summary(
                                    transcript_string,
                                    model=st.session_state[STATE_OPENAI_MODEL],
                                    target_language_code=target_lang_code # Pass the code
                                )

                                if speech_text_generated:
                                    status_placeholder_speech.success(f"{target_lang_name} summary generated.")
                                    text_saved = False
                                    audio_saved = False

                                    # 4. Save Text Summary
                                    try:
                                        with open(fixed_txt_path, "w", encoding='utf-8') as f:
                                            f.write(speech_text_generated)
                                        logger.info(f"Text file saved ({target_lang_code}): {fixed_txt_path}")
                                        text_saved = True
                                    except IOError as e:
                                        status_placeholder_speech.error(f"Failed to save text file: {e}")
                                        logger.error(f"IOError saving speech text to {fixed_txt_path}: {e}")
                                        # Proceed to audio generation? Maybe not if text failed. Let's stop here.
                                        raise # Re-raise to stop execution

                                    # 5. Generate Speech Audio
                                    status_placeholder_speech.info(f"Generating {target_lang_name} audio (TTS)...")
                                    audio_saved = generate_speech_audio(speech_text_generated, fixed_mp3_path)

                                    if audio_saved:
                                        status_placeholder_speech.success(f"{target_lang_name} audio generated. Ready for download.")
                                        # --- Store paths and filenames in session state ---
                                        st.session_state[STATE_SPEECH_TXT_PATH] = fixed_txt_path
                                        st.session_state[STATE_SPEECH_MP3_PATH] = fixed_mp3_path
                                        st.session_state[STATE_SPEECH_USER_TXT_FILENAME] = user_facing_txt_filename
                                        st.session_state[STATE_SPEECH_USER_MP3_FILENAME] = user_facing_mp3_filename
                                        st.session_state[STATE_SPEECH_READY] = True
                                        success_flag = True # Mark overall success
                                    else:
                                         status_placeholder_speech.error(f"Failed to generate {target_lang_name} audio file.")
                                         # Clean up text file if audio failed?
                                         if text_saved and os.path.exists(fixed_txt_path):
                                              try: os.remove(fixed_txt_path)
                                              except OSError: logger.warning(f"Could not remove text file {fixed_txt_path} after audio failure.")

                                else:
                                    status_placeholder_speech.error(f"LLM failed to generate {target_lang_name} speech summary.")
                            else:
                                status_placeholder_speech.warning("Transcript was empty, cannot generate speech.")
                        else:
                            # load_transcript shows error (e.g., no transcript) via st.error/warning
                            status_placeholder_speech.warning("Could not load transcript for speech generation.")

                    except Exception as e:
                        status_placeholder_speech.error(f"An error occurred: {e}")
                        logger.exception(f"Error in 'A brief speech' flow for {validated_url} ({target_lang_code}): {e}")
                        # Cleanup potentially created files on error
                        if fixed_txt_path and os.path.exists(fixed_txt_path):
                             try: os.remove(fixed_txt_path)
                             except OSError: pass
                        if fixed_mp3_path and os.path.exists(fixed_mp3_path):
                              try: os.remove(fixed_mp3_path)
                              except OSError: pass

                # --- Trigger a rerun AFTER the button logic finishes ---
                # This allows the download buttons (rendered below based on state) to appear/update.
                st.rerun()

            # Else: validate_youtube_url handled error display via st.sidebar.error

    with col3:
        if st.button("🔄 New Chat", key='yt_refresh_button', help="Clear YouTube Q&A history"):
            # Reset speech state on new chat? Optional, but maybe good practice.
            st.session_state[STATE_SPEECH_READY] = False
            st.session_state[STATE_SPEECH_TXT_PATH] = None
            st.session_state[STATE_SPEECH_MP3_PATH] = None
            # Clear status messages if needed
            status_placeholder_speech.empty()

            # Reset Q&A history
            st.session_state[STATE_YT_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]
            st.toast("YouTube chat history reset!", icon="🔄")
            st.rerun() # Rerun to clear messages display

    # --- Display Download Buttons based on Session State (AFTER columns) ---
    if st.session_state.get(STATE_SPEECH_READY, False):
        txt_path = st.session_state.get(STATE_SPEECH_TXT_PATH)
        mp3_path = st.session_state.get(STATE_SPEECH_MP3_PATH)
        txt_user_filename = st.session_state.get(STATE_SPEECH_USER_TXT_FILENAME, "speech.txt")
        mp3_user_filename = st.session_state.get(STATE_SPEECH_USER_MP3_FILENAME, "speech.mp3")
        current_lang_code = st.session_state.get(STATE_SPEECH_LANG_CODE, '')
        current_lang_name = SUPPORTED_SPEECH_LANGUAGES.get(current_lang_code, '')

        st.sidebar.markdown("---") # Separator
        st.sidebar.caption(f"Speech Downloads ({current_lang_name}):") # Header indicates language

        download_cols = st.sidebar.columns(2) # Place buttons side-by-side

        with download_cols[0]:
             if txt_path and os.path.exists(txt_path):
                 try:
                     with open(txt_path, "rb") as fp_txt: # Read as bytes for download button
                         st.download_button(
                             label="📜 Text (.txt)", # Icon/Shorter label
                             data=fp_txt.read(),
                             file_name=txt_user_filename,
                             mime="text/plain",
                             key=f"speech_txt_dl_btn_{current_lang_code}" # Language specific key
                         )
                 except Exception as e:
                     st.sidebar.error(f"Txt DL Error: {e}")
                     logger.error(f"Error creating text download button {txt_path}: {e}")
             else:
                  st.sidebar.caption("Text N/A") # Indicate if file missing

        with download_cols[1]:
             if mp3_path and os.path.exists(mp3_path):
                 try:
                     with open(mp3_path, "rb") as fp_mp3:
                         st.download_button(
                             label="🔊 Audio (.mp3)", # Icon/Shorter label
                             data=fp_mp3.read(),
                             file_name=mp3_user_filename,
                             mime="audio/mpeg",
                             key=f"speech_mp3_dl_btn_{current_lang_code}" # Language specific key
                         )
                         # Add audio player for convenience
                         st.audio(mp3_path, format='audio/mpeg')
                 except Exception as e:
                     st.sidebar.error(f"Audio DL Error: {e}")
                     logger.error(f"Error creating audio download button {mp3_path}: {e}")
             else:
                 st.sidebar.caption("Audio N/A") # Indicate if file missing


    # --- Display Currently Analyzed Video ---
    # Show video only if URL is valid
    if yt_chat_url_from_input and validate_youtube_url(yt_chat_url_from_input): # Check validity again before display
         st.sidebar.markdown("---")
         st.sidebar.caption("Currently Loaded Video:")
         # Use columns to control width/prevent excessive height
         vid_col1, vid_col2 = st.sidebar.columns([3, 1]) # Allocate more space to video
         with vid_col1:
              st.video(yt_chat_url_from_input)

    st.divider()

    # --- YouTube Download ---
    st.caption("📥 Download Video (MP4, Max 720p)")
    yt_dn_url = st.text_input(
        "Download YouTube Video (Max 720p)",
        placeholder='Paste video URL here...',
        label_visibility='collapsed',
        key='yt_download_url_input'
    )
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if st.button("🔗 Link (y2mate)", key='yt_y2mate_button', help="Get external download link (opens new tab)"):
            validated_dl_url = validate_youtube_url(yt_dn_url)
            if validated_dl_url:
                y2mate_link = generate_y2mate_link(validated_dl_url)
                if y2mate_link:
                    # Use markdown for clickable link, ensure it opens in new tab
                    st.write(f"➡️ [Open y2mate Download Page]({y2mate_link})", unsafe_allow_html=True)
                    # JS to open in new tab (more reliable than target="_blank" in markdown sometimes)
                    # components.html(f"<script>window.open('{y2mate_link}', '_blank');</script>", height=0, width=0)
                    st.caption("(Opens in new tab)")
                # else: generate_y2mate_link shows warning
            # Else: validate shows error
    with col_dl2:
        if st.button("⬇️ Download Direct", key='yt_direct_download_button', help="Download video directly using server resources"):
             validated_dl_url = validate_youtube_url(yt_dn_url)
             if validated_dl_url:
                 yt_download(validated_dl_url) # Handles download and provides button/error UI
             # Else: validate shows error


def render_document_sidebar():
    """Renders sidebar elements specific to the Document function."""
    st.caption("📄 Document Processing & Analysis")
    # --- Document Selection ---
    try:
        # List only .bin files (processed databases)
        doc_files = sorted([f for f in os.listdir(DOC_DATA_PATH) if f.endswith(".bin")]) # Sort list
        if not doc_files:
            st.info("No processed documents found. Upload a document to begin.")
            doc_files_options = ["No documents available"]
            current_selection_index = 0
        else:
            doc_files_options = doc_files
            # Try to find index of currently selected doc, default to 0 if not found/not set
            current_selection = st.session_state.get(STATE_SELECTED_DOC_PATH)
            current_filename = os.path.basename(current_selection) if current_selection else None
            try:
                 current_selection_index = doc_files.index(current_filename) if current_filename in doc_files else 0
            except ValueError:
                 current_selection_index = 0


        selected_doc_filename = st.selectbox(
            "Select Processed Document:",
            doc_files_options,
            index=current_selection_index, # Set default selection
            label_visibility="collapsed",
            key="doc_selector"
        )

        # Buttons for selected document actions
        col1, col2, col3 = st.columns(3)
        can_interact = selected_doc_filename != "No documents available"

        with col1:
             # Load button sets the selected path in session state
             if st.button("✅ Load", key="doc_load_button", disabled=not can_interact, help="Load selected document for Q&A"):
                 new_path = os.path.join(DOC_DATA_PATH, selected_doc_filename)
                 # Only update and rerun if selection actually changes
                 if st.session_state.get(STATE_SELECTED_DOC_PATH) != new_path:
                      st.session_state[STATE_SELECTED_DOC_PATH] = new_path
                      st.toast(f"Loaded '{selected_doc_filename}' for analysis.", icon="✅")
                      # Reset doc chat history when loading a new doc
                      st.session_state[STATE_DOC_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]
                      st.rerun() # Rerun to update main panel state display
                 else:
                      st.toast(f"'{selected_doc_filename}' is already loaded.", icon="ℹ️")


        with col2:
             # Delete button removes the .bin file
             if st.button("❌ Delete", key="doc_delete_button", disabled=not can_interact, help="Delete selected processed document"):
                 file_to_delete = os.path.join(DOC_DATA_PATH, selected_doc_filename)
                 try:
                     os.remove(file_to_delete)
                     st.success(f"Deleted '{selected_doc_filename}'.")
                     # Clear selection if the loaded file was deleted
                     if st.session_state.get(STATE_SELECTED_DOC_PATH) == file_to_delete:
                         st.session_state[STATE_SELECTED_DOC_PATH] = None
                         st.session_state[STATE_DOC_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}] # Also reset chat
                     st.rerun() # Rerun to update the file list and main panel
                 except OSError as e:
                     st.error(f"Error deleting file: {e}")


        with col3:
             # New Chat button resets message history for the currently loaded doc
             doc_loaded = bool(st.session_state.get(STATE_SELECTED_DOC_PATH))
             if st.button("🔄 New Chat", key="doc_new_chat_button", disabled=not doc_loaded, help="Clear Q&A history for the loaded document"):
                 st.session_state[STATE_DOC_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]
                 st.toast("Document chat history reset.", icon="🔄")
                 st.rerun() # Rerun to clear messages display

    except FileNotFoundError:
        st.error(f"Document directory not found: {DOC_DATA_PATH}")
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        logger.error(f"Error rendering document sidebar selection: {e}", exc_info=True)

    st.divider()

    # --- File Uploader ---
    st.caption("⬆️ Upload New File")
    uploaded_file = st.file_uploader(
        "Upload audio (mp3, m4a, wav) or document (pdf, txt, docx, md, html, csv)",
        type=["mp3", "m4a", "wav", "ogg", "flac", "pdf", "txt", "docx", "md", "html", "csv"], # Expanded types
        label_visibility="collapsed",
        key="doc_uploader"
    )

    # Use a separate button to trigger processing after upload
    # This prevents processing on every interaction with the uploader widget itself
    if uploaded_file is not None:
         if st.button(f"Process '{uploaded_file.name}'", key="process_upload_button"):
             # --- Prevent re-processing the same upload if desired (using name check) ---
             # Get the name of the file just uploaded
             current_upload_filename = uploaded_file.name
             # # Optional: Check against last processed name (commented out for now, process always allowed)
             # last_processed = st.session_state.get('last_processed_doc_upload_name', None)
             # if current_upload_filename == last_processed:
             #     st.info(f"'{current_upload_filename}' was already processed in this session.")
             #     return # Stop if already processed

             logger.info(f"Processing uploaded file: {current_upload_filename}")
             # Use a temporary directory or ensure unique temp names if parallel uploads were possible
             temp_file_path = os.path.join(DOC_DATA_PATH, f"temp_{current_upload_filename}")
             processing_success = False # Flag to track success

             # Show spinner during processing
             with st.spinner(f"Processing '{current_upload_filename}'... This may take a while."):
                 try:
                     # Save temporary file
                     with open(temp_file_path, "wb") as f:
                         f.write(uploaded_file.getvalue())
                     logger.info(f"Uploaded file saved temporarily to: {temp_file_path}")

                     # Process the saved file - Now returns True/False
                     # file_processing_pipeline handles its own st.sidebar messages
                     processing_result = file_processing_pipeline(temp_file_path, current_upload_filename)

                     if processing_result is True:
                          processing_success = True
                          # st.session_state['last_processed_doc_upload_name'] = current_upload_filename # Store if needed
                          # Success message handled by pipeline
                     else:
                          # Failure is handled within the pipeline (st.sidebar.error)
                          processing_success = False

                 except IOError as e:
                     logger.error(f"Error saving uploaded file {current_upload_filename}: {e}")
                     st.sidebar.error(f"Failed to save uploaded file: {e}") # Show error in sidebar
                     processing_success = False
                 except Exception as e:
                      logger.error(f"Error during upload processing trigger for {current_upload_filename}: {e}", exc_info=True)
                      st.sidebar.error(f"An error occurred initiating file processing: {e}") # Show error in sidebar
                      processing_success = False
                 # finally: # Cleanup is handled within the pipeline itself now

             # Rerun AFTER processing attempt to update the file list in the selectbox
             st.rerun()

# --- Remaining helper functions (render_wolfram_widget, reset_chat, display_messages) ---
# These functions remain largely the same as in the provided code,
# just ensure correct session state keys and error handling.

def render_wolfram_widget():
    """Renders a WolframAlpha search input and button in the sidebar."""
    with st.expander("🌟 **WolframAlpha Search** 🌟", expanded=False):
        query = st.text_input(
            "Enter WolframAlpha query:",
            key="wolfram_query_input", # Add a unique key
            label_visibility="collapsed", # Hide label if header is enough
            placeholder="e.g., population of France"
            )
        search_button = st.button("Go", key="wolfram_search_button")

        if search_button:
            if query:
                try:
                    encoded_query = quote_plus(query)
                    wolfram_url = f"https://www.wolframalpha.com/input/?i={encoded_query}"
                    logger.info(f"Opening WolframAlpha search: {wolfram_url}")

                    # Use Python's webbrowser module to open the URL
                    webbrowser.open_new_tab(wolfram_url)
                    st.toast(f"Opened search for '{query}' in new tab.", icon="✅")

                except Exception as e:
                    st.error(f"Could not open search link: {e}")
                    logger.error(f"Error opening WolframAlpha link for query '{query}': {e}", exc_info=True)
            else:
                st.toast("Please enter a query first.", icon="⚠️")

def reset_chat(template_name: str):
    """Resets chat history and applies the system prompt for the selected template."""
    try:
        # Ensure template name is valid before getting config
        from chat_config import TEMPLATE_CONFIGS # Import dict
        if template_name not in TEMPLATE_CONFIGS:
            logger.error(f"Invalid template name '{template_name}' during chat reset. Falling back to default.")
            template_name = "Agent Primrose" # Fallback to default

        config_func, _ = get_template_config(template_name) # Use the getter function
        start_content, _ = config_func()
        st.session_state[STATE_CHAT_MESSAGES] = [{"role": "system", "content": start_content}]
        # Reset other chat-related state
        st.session_state[STATE_FULL_RESPONSE] = ""
        st.session_state[STATE_MEME_CAPTION] = ""
        if 'image_to_process' in st.session_state:
            del st.session_state['image_to_process'] # Clear pending image
        logger.info(f"Chat reset with template: {template_name}")
    except ValueError as e: # Catch error from get_template_config if needed
        logger.error(f"Error getting config for template '{template_name}' during chat reset: {e}")
        st.error("Could not reset chat with the selected persona.")
        # Apply a basic default system prompt as a fallback
        st.session_state[STATE_CHAT_MESSAGES] = [{"role": "system", "content": "You are a helpful assistant."}]
    except Exception as e:
        logger.error(f"Unexpected error during chat reset: {e}", exc_info=True)
        st.error("An unexpected error occurred while resetting the chat.")
        st.session_state[STATE_CHAT_MESSAGES] = [{"role": "system", "content": "You are a helpful assistant."}]


def display_messages(messages: List[Dict[str, Any]], user_avatar: str, assistant_avatar: str):
    """Displays chat messages with appropriate avatars, handling complex content."""
    if not messages: # Handle case where messages might be empty
        logger.debug("Message list is empty, nothing to display.")
        return

    # Skip the system prompt (first message if it exists and is system)
    # Check if the first message exists and is a dictionary before accessing 'role'
    displayable_messages = []
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        displayable_messages = messages[1:]
    else:
        displayable_messages = messages

    for idx, message in enumerate(displayable_messages):
        # Basic validation
        if not isinstance(message, dict):
             logger.warning(f"Skipping invalid message format at index {idx}: {message}")
             continue
        role = message.get("role")
        content = message.get("content")

        # Use default avatar if role is unknown, though should be 'user' or 'assistant'
        avatar = user_avatar if role == "user" else assistant_avatar
        default_role_name = role if role in ["user", "assistant"] else "unknown"

        with st.chat_message(default_role_name, avatar=avatar):
            # Handle complex content (list with text/image dicts) vs simple string
            if isinstance(content, list):
                 for item in content:
                     if isinstance(item, dict):
                         item_type = item.get("type")
                         if item_type == "text":
                             # Basic sanitization for display
                             text_to_display = item.get("text", "").replace("~", "&#126;")
                             st.write(text_to_display, unsafe_allow_html=True) # Use markdown
                         elif item_type == "image_url":
                              img_url_data = item.get("image_url", {})
                              img_url = img_url_data.get("url")
                              if img_url and img_url.startswith("data:image"):
                                  st.image(img_url, width=150) # Display base64 image thumbnail
                                  # st.caption("Image provided by user")
                              elif img_url:
                                  st.image(img_url, width=150) # Display URL image thumbnail
                                  # st.caption("Image provided by user")
                              else:
                                   st.write("*(Image part missing URL)*")
                         else:
                              st.write(f"*(Unsupported content part type: {item_type})*")
                     else: # Should not happen if content structure is correct
                          logger.warning(f"Unexpected item type in content list: {type(item)}")
                          st.write(str(item))

            elif isinstance(content, str):
                # Basic sanitization for display
                st.write(content.replace("~", "&#126;"), unsafe_allow_html=True) # Use markdown
            elif content is None:
                st.write("*(Empty message)*") # Handle None content
            else:
                 logger.warning(f"Unexpected content type in message: {type(content)}")
                 st.write(str(content)) # Fallback for unexpected content types

# --- Main Application Logic (Tabs) ---

def handle_chat_tab():
    """Handles the logic and UI for the Chat tab."""
    st.subheader("💬🌸 Cozy Chat: Relax, Share & Discover Together")
    st.divider()
    with st.expander("🗨️ **Quick Guide**"):
        st.success("""
        - Enter your questions or topics to begin chatting with the selected persona.
        - **Special Prefixes:**
            - `.` : Use search-enhanced model (e.g., `.` latest AI news). *Note: May incur different costs.*
            - `/` : Clear the current chat and start fresh.
            - `!` : Fetch web content (e.g., `!` https://example.com).
        - **Image Chat (GPT-4.5-preview models):** Upload an image or provide a URL in the sidebar to discuss it. The image is included with your *next* message.
        """)

    # Ensure template is selected and messages initialized
    template_name = st.session_state.get('selected_template', "Agent Primrose")
    if not st.session_state.get(STATE_CHAT_MESSAGES):
        reset_chat(template_name) # Initialize with the current template's system prompt

    try:
        # Get avatar using the helper function/dict lookup
        from chat_config import TEMPLATE_CONFIGS
        avatar = TEMPLATE_CONFIGS.get(template_name, {}).get("avatar", "🤖") # Default avatar
    except Exception as e:
        logger.warning(f"Could not get avatar for template {template_name}: {e}. Using default.")
        avatar = "🤖" # Default avatar

    # Display chat history
    display_messages(st.session_state[STATE_CHAT_MESSAGES], "🌌", avatar)

    # Chat input guide text
    guide = "Enter your message..." # Default guide
    try:
        config_func, _ = get_template_config(template_name)
        _, guide = config_func()
    except ValueError:
        logger.error(f"Invalid template name '{template_name}' encountered when getting guide.")
    except Exception as e:
        logger.error(f"Error getting guide text for template '{template_name}': {e}")

    # Check for OpenAI client
    if not client:
         st.error("OpenAI client is not available. Chat functionality is disabled. Please check your API key setup.")
         return # Stop further processing in this tab


    if prompt := st.chat_input(guide, key="chat_input"):
        is_search_mode = False
        target_model = st.session_state[STATE_OPENAI_MODEL] # Start with the default selected model
        actual_prompt = prompt

        # --- Special Command Handling ---
        if prompt.startswith('!'):
            web_url = prompt[1:].strip()
            if web_url:
                with st.spinner(f"Trying to fetch content from {web_url} using BeautifulSoup..."):
                    web_content, status_code = fetch_main_content_bs4(web_url)

                # Check if BeautifulSoup successfully retrieved content
                if web_content and status_code == 200:
                    logger.info("Content fetched successfully with BeautifulSoup.")
                else:
                    logger.warning("BeautifulSoup failed or returned no content. Falling back to Selenium...")
                    st.warning("BeautifulSoup couldn't fetch content. Trying Selenium...")

                    with st.spinner(f"Fetching content from {web_url} using Selenium..."):
                        web_content, status_code = fetch_main_content_selenium(web_url)

                # After attempting both methods, evaluate final result
                if web_content and status_code == 200:
                    user_web_prompt = f"Analyze the following content fetched from URL: {web_url}\n\n---\n{web_content}\n---"
                    st.session_state[STATE_CHAT_MESSAGES].append({"role": "user", "content": user_web_prompt})
                    # Add a clarifying assistant message
                    st.session_state[STATE_CHAT_MESSAGES].append({"role": "assistant", "content": f"I have received the content from {web_url}. How can I help you analyze or discuss it?"})
                    st.rerun()  # Automatically trigger assistant response
                elif status_code:
                    st.error(f"🚨 Failed to fetch content (Status Code: {status_code}). Please verify the URL or try again later.")
                else:
                    st.error("🚨 Failed to fetch content. The website might be blocking requests or the URL might be invalid.")
            else:
                st.warning("Please provide a URL after the '!'")
            return  # Stop further processing after '!'

        elif prompt.startswith('/'):
             reset_chat(template_name)
             st.info("Chat cleared. Start your new conversation below!")
             st.rerun() # Rerun to reflect cleared state immediately
             return # Exit processing

        elif prompt.startswith('.'):
             # Define the search preview model - adjust if OpenAI changes the name
             search_preview_model = "gpt-4o-search-preview" # Currently gpt-4o supports browsing/search
             # Check if current model *can* be switched or is already the right one
             if target_model != search_preview_model:
                 # For simplicity, always switch if prefix is used. Add checks if needed.
                 is_search_mode = True
                 st.session_state['original_model'] = target_model # Store original
                 target_model = search_preview_model # Use the search-capable model
                 logger.info(f"Switching to search-enabled model ({target_model}) for this query.")
             else:
                  is_search_mode = True # Already using a capable model
                  logger.info(f"Using current model ({target_model}) with search capabilities.")

             actual_prompt = prompt[1:].strip()
             if not actual_prompt:
                 st.warning("Please enter a search query after '.'")
                 # Revert model if we switched unnecessarily
                 if 'original_model' in st.session_state:
                     target_model = st.session_state['original_model']
                     del st.session_state['original_model']
                 return # Stop processing

        else: # Regular prompt
             is_search_mode = False
             actual_prompt = prompt

        # Prepare user message content (handling potential image)
        user_message_content: Any = actual_prompt
        image_to_send = st.session_state.get('image_to_process')

        # Include image ONLY if available AND the target model is vision-capable
        if image_to_send and target_model in ['gpt-4.5-preview', 'o3-mini']:
            try:
                # Convert image to base64
                # Ensure format compatibility or convert
                if image_to_send.format not in ["PNG", "JPEG", "GIF", "WEBP"]:
                     logger.info(f"Converting image from {image_to_send.format} to PNG for API.")
                     with BytesIO() as output:
                          # Convert to RGB before saving as PNG if it's RGBA or other modes
                          img_rgb = image_to_send.convert('RGB')
                          img_rgb.save(output, format="PNG")
                          png_data = output.getvalue()
                     base64_image = image_to_base64(png_data, is_bytes=True) # Use updated function
                     mime_type = "image/png"
                else:
                     # Directly convert compatible formats
                     img_bytesio = BytesIO()
                     image_to_send.save(img_bytesio, format=image_to_send.format)
                     base64_image = image_to_base64(img_bytesio.getvalue(), is_bytes=True) # Use updated function
                     mime_type = f"image/{image_to_send.format.lower()}"


                user_message_content = [
                    {"type": "text", "text": actual_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
                logger.info("Image included in the user message.")
                # Clear image from state AFTER successfully adding it to the message
                if 'image_to_process' in st.session_state:
                     del st.session_state['image_to_process']
            except Exception as e:
                logger.error(f"Failed to process image for chat: {e}", exc_info=True)
                st.error("Could not include the image in the prompt. Sending text only.")
                user_message_content = actual_prompt # Fallback to text only
                # Clear image from state even on failure
                if 'image_to_process' in st.session_state:
                     del st.session_state['image_to_process']


        # Add user message to history (important step before API call)
        st.session_state[STATE_CHAT_MESSAGES].append({"role": "user", "content": user_message_content})

        # --- Display user message immediately ---
        # This mirrors the user input instantly before the assistant responds.
        with st.chat_message("user", avatar="🌌"):
            if isinstance(user_message_content, list):
                for item in user_message_content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            st.write(item.get("text", ""), unsafe_allow_html=True)
                        elif item_type == "image_url":
                            img_url_data = item.get("image_url", {})
                            img_url = img_url_data.get("url")
                            if img_url and img_url.startswith("data:image"):
                                st.image(img_url, width=150) # Display the image that was sent
            elif isinstance(user_message_content, str):
                st.write(user_message_content, unsafe_allow_html=True)
            else:
                st.write(str(user_message_content))
        # --- End immediate display ---


        # --- Assistant Response Generation ---
        with st.chat_message("assistant", avatar=avatar):
            message_placeholder = st.empty()
            try:
                # Prepare messages for API call (include history)
                # Apply Meme Maestro modification if needed
                messages_for_api = st.session_state[STATE_CHAT_MESSAGES]
                if template_name == "Meme Maestro" and messages_for_api[-1]["role"] == "user":
                     # Ensure we don't modify the original list in session state directly if possible
                     # Create a deep copy if necessary, or modify carefully
                     last_user_msg_content = messages_for_api[-1]["content"] # Get content ref
                     meme_instruction = "\nMeme Maestro: Sends a meme of {...} with the caption {...}"

                     # Modify based on content type (string or list)
                     if isinstance(last_user_msg_content, str):
                         # Create a *new* message object to avoid modifying state directly before API call
                         modified_content = last_user_msg_content + meme_instruction
                         messages_for_api = messages_for_api[:-1] + [{"role": "user", "content": modified_content}]
                     elif isinstance(last_user_msg_content, list):
                          # Modify the text part within the list (carefully)
                          # Create a *copy* of the list and dicts to avoid side effects
                          import copy
                          copied_content_list = copy.deepcopy(last_user_msg_content)
                          text_part_modified = False
                          for item in copied_content_list:
                               if item.get("type") == "text":
                                    item["text"] += meme_instruction
                                    text_part_modified = True
                                    break
                          if text_part_modified:
                               # Use the modified list in a *new* message object
                               messages_for_api = messages_for_api[:-1] + [{"role": "user", "content": copied_content_list}]
                          # else: instruction not added if no text part found (shouldn't happen)

                # Define API parameters
                api_params = {
                    "model": target_model, # Use the potentially switched model
                    "messages": messages_for_api,
                    "stream": True,
                }
                # Add temp/top_p/max_tokens conditionally based on model/mode
                if not is_search_mode and target_model in ["gpt-4.5-preview"]: # Example condition
                    api_params["temperature"] = 0.7
                    # api_params["top_p"] = 0.9 # Optional
                    # api_params["max_tokens"] = st.session_state[STATE_TOKEN_SIZE] # Example if needed

                # Call OpenAI API
                stream = client.chat.completions.create(**api_params)

                # Parse stream and display response
                # Reset full response state key before streaming new response
                st.session_state[STATE_FULL_RESPONSE] = ""
                stream_generator = parse_stream(stream)
                # Display stream in the placeholder
                message_placeholder.write_stream(stream_generator)

                # Get the final accumulated response from the session state updated by parse_stream
                full_response = st.session_state.get(STATE_FULL_RESPONSE, "")

                # Add assistant's *final* full response to history (AFTER generation)
                if full_response or full_response == "": # Add even if empty if intended
                     st.session_state[STATE_CHAT_MESSAGES].append({"role": "assistant", "content": full_response})
                else:
                      logger.warning("Assistant response was empty after streaming.")
                      # Optionally add a placeholder message to history if needed
                      # st.session_state[STATE_CHAT_MESSAGES].append({"role": "assistant", "content": "[No response received]"})


                # --- Post-Response Actions ---
                if template_name == "Meme Maestro" and full_response:
                    st.session_state[STATE_MEME_CAPTION] = full_response # Store caption
                    with st.spinner("Generating meme... ✨"):
                            gen_url = find_and_generate_meme(full_response)
                    if gen_url:
                        display_meme(gen_url, full_response) # Display meme below the text
                    # else: find_and_generate_meme handles st.error

                # Limit chat history length
                # message_limits = {"Translator": 4} # System + 3 turns = 7 messages (3 Q+A pairs)
                # limit = message_limits.get(template_name, 20) # Default: System + ~10 turns = 21 messages
                max_history_length = 20 # Example: Keep system prompt + last 9 Q&A pairs
                if len(st.session_state[STATE_CHAT_MESSAGES]) > max_history_length:
                     # Keep the first (system) message and the last (limit - 1) messages
                     st.session_state[STATE_CHAT_MESSAGES] = [st.session_state[STATE_CHAT_MESSAGES][0]] + st.session_state[STATE_CHAT_MESSAGES][-(max_history_length-1):]
                     logger.info(f"Chat history trimmed to {max_history_length} messages.")

            except Exception as e:
                logger.exception(f"Error during chat completion: {e}")
                error_message = f"An error occurred: {e}"
                message_placeholder.error(error_message)
                # Add error message to chat history for context
                st.session_state[STATE_CHAT_MESSAGES].append({"role": "assistant", "content": f"Error: {error_message}"})


            finally:
                 # Reset model back from search preview if it was used
                 if 'original_model' in st.session_state:
                     st.session_state[STATE_OPENAI_MODEL] = st.session_state['original_model']
                     del st.session_state['original_model'] # Clean up temporary state
                     logger.info(f"Reset model back to: {st.session_state[STATE_OPENAI_MODEL]}")

                 # Clear full response state key for next turn
                 if STATE_FULL_RESPONSE in st.session_state:
                      st.session_state[STATE_FULL_RESPONSE] = ""

                 # No st.rerun() needed here typically, Streamlit handles the updates

        # Add a small delay before potentially rerunning if needed elsewhere? Generally avoid reruns here.


def handle_document_tab():
    """Handles the logic and UI for the Document Analysis tab."""
    st.subheader("📚✨ Document Hub: Explore Knowledge & Insights")
    st.divider()
    with st.expander("📖 **Quick Guide**"):
        st.info("""
        - **Upload:** Use the sidebar to upload a new document (text, PDF, audio). Click 'Process' to create a searchable database.
        - **Select:** Choose a previously processed document from the dropdown in the sidebar.
        - **Load:** Click 'Load' to activate the selected document for Q&A.
        - **Ask:** Enter your questions about the loaded document's content below.
        """)

    # Check for OpenAI client (needed for embeddings and Q&A)
    if not client:
         st.error("OpenAI client is not available. Document analysis functionality is disabled. Please check your API key setup.")
         return

    # Display loaded document status
    selected_path = st.session_state.get(STATE_SELECTED_DOC_PATH)
    doc_ready = selected_path and os.path.exists(selected_path)

    if doc_ready:
        base_name = os.path.basename(selected_path)
        display_name = base_name.replace('.bin', '') if base_name else "document"
        st.success(f"Ready to answer questions about: **{display_name}**")
    else:
        st.info("Please upload and process a document, or load an existing one from the sidebar to begin analysis.")

    # Display chat history for documents first
    display_messages(st.session_state.get(STATE_DOC_MESSAGES, []), "🌌", "📑")

    # Chat input - Only enable if a document is loaded
    if doc_ready:
        base_name = os.path.basename(selected_path)
        display_name = base_name.replace('.bin', '') if base_name else "document"
        if prompt := st.chat_input(f"Ask about {display_name}:", key="doc_chat_input", disabled=not doc_ready):
            st.session_state[STATE_DOC_MESSAGES].append({"role": "user", "content": prompt})

            # Display user message immediately
            with st.chat_message("user", avatar="🌌"):
                 st.write(prompt) # Use markdown

            # Assistant response generation
            with st.chat_message("assistant", avatar="📑"):
                message_placeholder = st.empty() # Create placeholder HERE
                full_response = ""
                sources = None

                with st.spinner("Analyzing document and formulating response..."):
                    try:
                        # Load the serialized DB
                        with open(selected_path, "rb") as file:
                            serialized_db = pickle.load(file)

                        # Define retrieval parameters
                        k_retrieval = 5 # Example: retrieve top 5 docs
                        model_name = st.session_state[STATE_OPENAI_MODEL]

                        # Call the updated function, passing the placeholder
                        full_response, sources = get_response_from_query(
                            serialized_db,
                            prompt,
                            k=k_retrieval,
                            model_name=model_name,
                            message_placeholder=message_placeholder # Pass placeholder
                        )

                        # Add the final response (collected by handler) to history
                        # Ensure response is added even if it's an error message from the function
                        st.session_state[STATE_DOC_MESSAGES].append({"role": "assistant", "content": full_response or "[No response or error]"})
                        
                        # Optionally display sources if successful response and sources exist
                        if sources and not (full_response or "").startswith("Error"):
                            with st.expander("Sources Used"):
                                for i, source in enumerate(sources):
                                    content_to_display = getattr(source, 'page_content', 'Source content not available.')
                                    # Display limited preview
                                    preview = (content_to_display[:200] + '...') if len(content_to_display) > 200 else content_to_display
                                    st.caption(f"Source {i+1}:")
                                    st.write(f"> {preview}") # Blockquote style

                    except FileNotFoundError:
                        err_msg = f"Document DB file not found: {selected_path}"
                        logger.error(err_msg)
                        message_placeholder.error(err_msg)
                        st.session_state[STATE_DOC_MESSAGES].append({"role": "assistant", "content": err_msg})
                    except pickle.UnpicklingError:
                        err_msg = f"Failed to load document DB (corrupted?): {selected_path}"
                        logger.error(err_msg)
                        message_placeholder.error(err_msg)
                        st.session_state[STATE_DOC_MESSAGES].append({"role": "assistant", "content": err_msg})
                    except Exception as e:
                        err_msg = f"An external error occurred during document query: {e}"
                        logger.exception(f"Error querying document DB {selected_path}: {e}")
                        message_placeholder.error(err_msg)
                        st.session_state[STATE_DOC_MESSAGES].append({"role": "assistant", "content": err_msg})

            # No explicit rerun needed here, streaming handles updates.

    # (display_messages was moved up)

def handle_youtube_tab():
    """Handles the logic and UI for the YouTube Analysis tab."""
    st.subheader("🎬✨ YouTube Explorer: Discover & Analyze Videos")
    st.divider()

    # Check for OpenAI client (needed for Q&A and Speech Summary)
    if not client:
         st.error("OpenAI client is not available. YouTube analysis features (Q&A, Speech) are disabled. Please check your API key setup.")
         # Optionally disable parts of the UI or just let functions fail later

    # --- Display Search Results First (if they exist) ---
    search_results = st.session_state.get('yt_search_results')
    if search_results:
        st.subheader("Search Results")
        cols_per_row = 4
        num_videos = len(search_results)
        num_rows = (num_videos + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            row_results = search_results[i * cols_per_row:(i + 1) * cols_per_row]
            cols = st.columns(cols_per_row)
            for j, result in enumerate(row_results):
                 if j < len(cols):
                    with cols[j]:
                        try:
                            video_id = result.get('id', '')
                            video_url = result.get('url', '')
                            video_title = result.get('title', 'N/A')
                            # Sanitize title for display
                            display_title = video_title.replace("[", "\\[").replace("]", "\\]")

                            # Show thumbnail or placeholder if video fails
                            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                            st.image(thumbnail_url, caption=display_title, use_container_width=True)
                            # st.video(video_url) # Using video can be slow/resource intensive

                            # Button to select this video for analysis/speech
                            button_key = f"select_{video_id}"
                            if st.button("Select this video", key=button_key, help=f"Load '{display_title}' for analysis/speech"):
                                # Update the chat URL input field
                                st.session_state[STATE_YT_CHAT_URL_INPUT] = video_url
                                # Reset state for the new video
                                st.session_state[STATE_YT_DB_READY] = False
                                st.session_state[STATE_SPEECH_READY] = False
                                st.session_state[STATE_YT_MESSAGES] = [{"role": "system", "content": "Always be truthful. Let's think step by step."}]
                                # st.session_state['yt_search_results'] = None # Clear search results after selection
                                st.toast(f"Selected '{display_title}'. Use sidebar options.", icon="🎬")
                                st.rerun() # Rerun to update sidebar input and clear search results view

                        except Exception as e:
                            logger.error(f"Error displaying search result {result.get('id')}: {e}")
                            cols[j].error("Error loading video preview.")

        st.divider()
        if st.button("Clear Search Results", key="clear_yt_search"):
             st.session_state['yt_search_results'] = None
             st.rerun()

    # --- YouTube Analysis/Q&A Section ---
    # Display chat history first
    display_messages(st.session_state.get(STATE_YT_MESSAGES, []), "🌌", "📺")

    # Determine if the analysis section should be expanded
    yt_db_is_ready = st.session_state.get(STATE_YT_DB_READY, False)
    # Expand if DB is ready OR if there are messages (indicating previous interaction)
    expand_analysis = yt_db_is_ready or len(st.session_state.get(STATE_YT_MESSAGES, [])) > 1

    with st.expander("▶️ **Video Analysis & Q&A**", expanded=expand_analysis):
        # Check if YouTube DB processing was successful *and* the file exists
        yt_db_file = os.path.join(YOUTUBE_DB_PATH, YOUTUBE_DB_FILENAME)
        if yt_db_is_ready and os.path.exists(yt_db_file):
            current_video_url = st.session_state.get(STATE_YT_CHAT_URL_INPUT)
            st.caption(f"✅ Analysis ready for: {current_video_url or 'current video'}")
        else: # DB not ready
            if st.session_state.get(STATE_YT_CHAT_URL_INPUT): # If a URL is entered but not ready
                 st.warning("Video not prepared for analysis. Please click 'Prepare Analysis' in the sidebar.")
            else: # No URL entered
                 st.info("Use the sidebar to search for a video or paste a URL, then click 'Prepare Analysis' to enable Q&A.")
    
    # Chat input for Q&A - Only available when DB is ready
    if client: # Check again if client is available for Q&A
        if prompt := st.chat_input("Ask questions about the video transcript:", key="yt_chat_input", disabled=not yt_db_is_ready):
            st.session_state[STATE_YT_MESSAGES].append({"role": "user", "content": prompt})

            # Display user message immediately
            with st.chat_message("user", avatar="🌌"):
                st.write(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="📺"):
                message_placeholder = st.empty() # Placeholder for streaming
                full_response = ""
                sources = None
                with st.spinner("Analyzing transcript and formulating response..."):
                    try:
                        with open(yt_db_file, "rb") as file:
                            serialized_db = pickle.load(file)

                        k_retrieval = 5 # Example k value
                        model_name = st.session_state[STATE_OPENAI_MODEL]

                        full_response, sources = get_response_from_query(
                            serialized_db,
                            prompt,
                            k=k_retrieval,
                            model_name=model_name,
                            message_placeholder=message_placeholder
                        )

                        # Add final response to history
                        st.session_state[STATE_YT_MESSAGES].append({"role": "assistant", "content": full_response or "[No response or error]"})

                        # Optional: Display sources
                        if sources and not (full_response or "").startswith("Error"):
                            with st.expander("Sources Used"):
                                # ... (source display logic as in document tab) ...
                                for i, source in enumerate(sources):
                                    content_to_display = getattr(source, 'page_content', 'Source content not available.')
                                    preview = (content_to_display[:200] + '...') if len(content_to_display) > 200 else content_to_display
                                    st.caption(f"Source {i+1}:")
                                    st.write(f"> {preview}")

                    except Exception as e:
                        logger.exception(f"Error querying YouTube DB: {e}")
                        err_msg = f"An external error occurred during Q&A: {e}"
                        message_placeholder.error(err_msg)
                        st.session_state[STATE_YT_MESSAGES].append({"role": "assistant", "content": err_msg})

    else:
        st.warning("OpenAI client not available, Q&A disabled.")


# --- Main Application Execution ---
def main():
    """Main function to run the Streamlit application."""
    # Initialize state ONCE per session at the very beginning
    if 'app_initialized' not in st.session_state:
         initialize_session_state()

    setup_page_config()
    # Don't initialize state again here, it should persist

    selected_model, selected_function = render_sidebar()

    logger.info(f"App Render. Model: {selected_model}, Function: {selected_function}, YT Ready: {st.session_state.get(STATE_YT_DB_READY)}, Doc Path: {st.session_state.get(STATE_SELECTED_DOC_PATH)}")

    # Display the appropriate tab based on function selection
    if selected_function == 'Chat':
        handle_chat_tab()
    elif selected_function == 'Document':
        handle_document_tab()
    elif selected_function == 'Youtube':
        handle_youtube_tab()
    else:
        st.error(f"Invalid function selected: {selected_function}") # Should not happen

if __name__ == "__main__":
    # Check for API key presence early
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
         logger.critical("OPENAI_API_KEY environment variable not set at startup.")
         # Displaying error here might be too early for Streamlit, handled in modules/tabs
    else:
         logger.info("OpenAI API Key found.")

    main()

# --- END OF REFACTORED FILE PrimGPT.py ---
