import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import time

import streamlit as st
import dill as pickle
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from youtubesearchpython import VideosSearch
from yt_dlp import YoutubeDL
import yt_dlp
import re 

# Local imports
from utils import get_youtube_video_id, validate_youtube_url, safe_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
YOUTUBE_DB_PATH = "./youtube_data/"
YOUTUBE_DB_FILENAME = "script_db.bin"
YOUTUBE_DOWNLOAD_PATH = "./youtube_downloads" # Changed back to match previous examples
# YOUTUBE_DOWNLOAD_TMPL = "downloaded_%(ext)s" # Template for downloaded files

# Ensure directories exist
os.makedirs(YOUTUBE_DB_PATH, exist_ok=True)
os.makedirs(YOUTUBE_DOWNLOAD_PATH, exist_ok=True) # Use consistent path variable

# --- Transcript & DB Functions ---

# --- MODIFIED load_transcript ---
def load_transcript(video_url: str, languages: List[str] = None, translation: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Loads YouTube transcript, trying preferred languages.
    Optionally translates to the specified language if different from the fetched one.
    """
    languages = languages if languages else ["en", "ko", "ja"]  # Default preferred languages (English first)
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        logger.error(f"Invalid YouTube URL for transcript: {video_url}")
        st.warning("Invalid YouTube URL provided.")
        return None

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_langs = [t.language_code for t in transcript_list]
        logger.info(f"Available transcript languages for {video_id}: {available_langs}")

        transcript = None
        # Try preferred languages first
        for lang in languages:
            if lang in available_langs:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    logger.info(f"Found transcript in preferred language: {lang}")
                    break
                except NoTranscriptFound: # Should not happen if lang in available_langs, but defensive check
                    logger.debug(f"No transcript found for language even though listed: {lang}")
                    continue

        # If no preferred language found, try the first available generated language
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(languages)
                logger.warning(f"No manually created transcript in preferred languages found. Using generated: {transcript.language}")
            except NoTranscriptFound:
                 logger.debug(f"No generated transcript found in preferred languages: {languages}")
                 # If still no transcript, try ANY available language as a last resort
                 if available_langs:
                    first_lang = available_langs[0]
                    try:
                        transcript = transcript_list.find_transcript([first_lang])
                        logger.warning(f"No preferred language transcript found. Using first available: {first_lang}")
                    except NoTranscriptFound: # Should not happen if available_langs is not empty
                        logger.error(f"Could not find transcript for supposedly available language: {first_lang}")
                        st.error("Failed to retrieve transcript even for available languages.")
                        return None
                 else: # No transcripts at all
                      logger.error(f"No transcripts found for video {video_id}.")
                      st.error("No transcripts available for this video (manual or generated).")
                      return None


        # Check if a transcript was found at all (should be covered above, but double-check)
        if not transcript:
            logger.error(f"No transcripts found for video {video_id} after all checks.")
            st.error("No transcripts available for this video.")
            return None

        # --- MODIFIED Translation Logic ---
        # Translate ONLY if a specific 'translation' language is provided AND it's different from the fetched transcript's language
        if translation and transcript.language != translation:
            try:
                # Check if the target language is available for translation
                if transcript.is_translatable and translation in [t.language_code for t in transcript.translation_languages]:
                    transcript = transcript.translate(translation)
                    logger.info(f"Transcript translated to: {translation}")
                else:
                    logger.warning(f"Transcript language {transcript.language} cannot be translated to {translation}. Using original.")
                    st.warning(f"Cannot translate from '{transcript.language}' to '{translation}'. Using original language.")
            except Exception as e:
                logger.error(f"Failed to translate transcript to {translation}: {e}")
                st.warning(f"Could not translate transcript to {translation}. Using original language: {transcript.language}")
        # --- END MODIFIED Translation Logic ---


        fetched_transcript = transcript.fetch()
        if not fetched_transcript:
            logger.warning(f"Fetched transcript for {video_id} is empty.")
            st.warning("The video transcript appears to be empty.")
            return [] # Return empty list, not None

        logger.info(f"Transcript loaded successfully for {video_id}. Language: {transcript.language}")
        return fetched_transcript

    except (NoTranscriptFound, TranscriptsDisabled):
        # This might be caught earlier, but keep as fallback
        logger.error(f"No transcript available or transcripts are disabled for video {video_id}.")
        st.error("Transcripts are disabled or unavailable for this video.")
        return None
    except VideoUnavailable:
        logger.error(f"Video {video_id} is unavailable.")
        st.error("This video is unavailable (private, deleted, etc.).")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error loading transcript for {video_id}: {e}")
        st.error(f"An unexpected error occurred while fetching the transcript: {e}")
        return None
# --- END MODIFIED load_transcript ---

def optimize_chunk_parameters(transcript_text: str,
                              default_chunk_size: int = 1000,
                              default_overlap: int = 100,
                              desired_chunks: int = 10,
                              min_chunk_size: int = 300,
                              min_overlap: int = 30) -> Tuple[int, int]:
    """Dynamically optimizes chunk size and overlap based on text length."""
    total_chars = len(transcript_text)
    if total_chars == 0 or desired_chunks <= 0:
        return default_chunk_size, default_overlap

    estimated_chunk_size = max(min_chunk_size, total_chars // desired_chunks)
    optimized_chunk_size = min(default_chunk_size, estimated_chunk_size)
    optimized_overlap = max(min_overlap, int(optimized_chunk_size * 0.1))

    logger.info(f"Optimized chunk size: {optimized_chunk_size}, overlap: {optimized_overlap} for text length {total_chars}")
    return optimized_chunk_size, optimized_overlap

def create_db_from_youtube_video_url(video_url: str,
                                     output_path: str = YOUTUBE_DB_PATH,
                                     output_filename: str = YOUTUBE_DB_FILENAME) -> bool:
    """Creates and saves a FAISS database from a YouTube transcript."""
    # Load transcript without forcing translation initially
    transcript_data = load_transcript(video_url, translation=None)
    if transcript_data is None:
        # load_transcript already showed error/warning
        return False

    if not transcript_data:
        st.warning("Transcript is empty, cannot create database.")
        return False

    full_transcript_text = " ".join(entry.get("text", "") for entry in transcript_data)
    if not full_transcript_text.strip():
        st.warning("Transcript contains no text content, cannot create database.")
        return False

    try:
        # Ensure embeddings can be created (requires API key)
        embeddings = OpenAIEmbeddings()
    except Exception as e:
         logger.error(f"Failed to initialize OpenAIEmbeddings: {e}")
         st.error(f"Failed to initialize embeddings. Check API key and connectivity. Cannot create DB for {video_url}")
         return False

    try:
        chunk_size, chunk_overlap = optimize_chunk_parameters(full_transcript_text)
        # Use video_url as source metadata
        document = Document(page_content=full_transcript_text, metadata={"source": video_url})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents([document])
        if not docs:
            logger.error("Text splitting resulted in zero documents.")
            st.error("Failed to split the transcript into processable chunks.")
            return False

        logger.info(f"Transcript split into {len(docs)} documents.")
        db = FAISS.from_documents(docs, embeddings)
        serialized_db = db.serialize_to_bytes()
        logger.info("FAISS database created.")

        db_filepath = os.path.join(output_path, output_filename)
        with open(db_filepath, "wb") as f:
            pickle.dump(serialized_db, f)

        logger.info(f"Database saved successfully to {db_filepath}")
        return True

    except Exception as e:
        logger.exception(f"Failed to create or save FAISS database for {video_url}: {e}")
        st.error(f"Error creating transcript database: {e}")
        return False

# --- ADD NEW FUNCTION for getting YouTube Title ---
def get_youtube_video_title(video_url: str) -> Optional[str]:
    """Fetches the title of a YouTube video using yt-dlp."""
    validated_url = validate_youtube_url(video_url) # Use validated URL
    if not validated_url:
        # Validation function already shows error
        return None

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True, # Don't download the video
        'force_generic_extractor': False, # Use YouTube specific extractor
    }
    try:
        logger.info(f"Fetching title for URL: {validated_url}")
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(validated_url, download=False)
            title = info_dict.get('title')
            if title:
                logger.info(f"Successfully fetched title: {title}")
                return title
            else:
                logger.warning(f"Could not extract title for {validated_url}")
                st.warning("Could not fetch the video title.")
                return None
    except yt_dlp.utils.DownloadError as e:
        # Handle specific yt-dlp errors, e.g., video unavailable
        error_message = str(e)
        logger.error(f"yt-dlp error fetching title for {validated_url}: {error_message}", exc_info=False)
        st.error(f"Could not fetch video info (maybe private or deleted?): {error_message}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching title for {validated_url}: {e}")
        st.error(f"An unexpected error occurred while fetching video title: {e}")
        return None
# --- END NEW FUNCTION ---

def yt_ready(yt_chat_url: str) -> Optional[bool]: # Return bool or None on error
    """Prepares the YouTube transcript database for querying."""
    db_filepath = os.path.join(YOUTUBE_DB_PATH, YOUTUBE_DB_FILENAME)
    if os.path.exists(db_filepath):
        try:
            os.remove(db_filepath)
            logger.info("Removed previous YouTube DB file.")
        except OSError as e:
            logger.error(f"Error removing previous DB file {db_filepath}: {e}")
            st.error("Could not clear the previous transcript data. Please check file permissions.")
            return None # Indicate failure

    # st.info("Processing YouTube video transcript...") # Moved inside create_db
    success = create_db_from_youtube_video_url(yt_chat_url, YOUTUBE_DB_PATH, YOUTUBE_DB_FILENAME)
    if success:
        # st.success("YouTube video is ready for questions!") # Let caller handle success message
        return True
    else:
        # create_db function already showed error
        # st.error("Failed to process YouTube video. Cannot proceed with questions.")
        return False

# --- YouTube Search & Download Functions ---
def yt_search(query: str, limit: int = 8):
    """Searches YouTube and stores results in session state."""
    if not query:
        st.warning("Please enter a search query.")
        st.session_state['yt_search_results'] = None
        return

    st.session_state['yt_search_results'] = None
    try:
        st.info(f"Searching YouTube for: '{query}'...")
        videos_search = VideosSearch(query, limit=limit)
        response = videos_search.result()
        results = response.get('result', [])

        if not results:
            st.warning("No videos found for your query.")
            return

        search_results_data = []
        for item in results:
            video_id = item.get('id')
            if video_id:
                search_results_data.append({
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'title': item.get('title', 'No Title'),
                    'id': video_id
                })

        st.session_state['yt_search_results'] = search_results_data
        logger.info(f"Stored {len(search_results_data)} search results in session state.")

    except Exception as e:
        logger.exception(f"Error during YouTube search for '{query}': {e}")
        st.error(f"An error occurred during YouTube search: {e}")
        st.session_state['yt_search_results'] = None

def generate_y2mate_link(video_url: str) -> Optional[str]:
    """Generates a y2mate link from a YouTube URL."""
    video_id = get_youtube_video_id(video_url)
    if video_id:
        return f"https://www.y2mate.com/youtube/{video_id}"
    else:
        st.sidebar.warning("Invalid YouTube URL for y2mate link.")
        return None

# --- MODIFIED yt_download Function (Overwrite Method) ---
def yt_download(video_url: str):
    """
    Downloads a YouTube video using yt-dlp, overwriting a fixed filename.
    Provides a download button with the original title.
    """
    validated_url = validate_youtube_url(video_url)
    if not validated_url:
        # st.sidebar.error("Invalid YouTube URL provided.") # validate_youtube_url shows error
        return

    # Use consistent download path
    os.makedirs(YOUTUBE_DOWNLOAD_PATH, exist_ok=True)

    # --- Define FIXED filename for overwriting ---
    fixed_filename_base = "downloaded_video" # Base name without extension

    # Prepare UI elements
    status_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    download_button_placeholder = st.sidebar.empty()

    status_text.info(f"Preparing download for: {validated_url}")

    def progress_hook(d):
        # --- Progress Hook Logic (remains the same) ---
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded_bytes = d.get('downloaded_bytes')
            if total_bytes and downloaded_bytes:
                percentage = int((downloaded_bytes / total_bytes) * 100)
                progress_bar.progress(percentage)
                speed = d.get('speed')
                eta = d.get('eta')
                speed_str = f"{speed / 1024 / 1024:.2f} MiB/s" if speed else "N/A"
                eta_str = f"{eta}s" if eta else "N/A"
                status_text.info(f"Downloading... {percentage}% ({speed_str}, ETA: {eta_str})")
            else: # Handle cases where bytes might not be available initially
                progress_bar.progress(0) # Show starting progress
                status_text.info("Downloading... (progress details unavailable yet)")

        elif d['status'] == 'finished':
            progress_bar.progress(100)
            logger.info(f"Download hook reported component finished: {d.get('filename', 'N/A')}")
            # Update status *before* verification sleep
            status_text.info(f"Processing download...") # Use info for processing step

        elif d['status'] == 'error':
            error_msg = d.get('error', 'Unknown download error')
            status_text.error(f"Download error: {error_msg}")
            progress_bar.empty()

    # Define yt-dlp options - Use the fixed base filename
    output_template = os.path.join(YOUTUBE_DOWNLOAD_PATH, f'{fixed_filename_base}.%(ext)s') # Use fixed name

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]', # Limit resolution
        'outtmpl': output_template, # Use the fixed filename template
        'progress_hooks': [progress_hook],
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4', # Output format (determines extension)
        'overwrites': True, # Explicitly allow overwriting
        'postprocessors': [{ # Ensure ffmpeg location is specified if needed
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        # Add ffmpeg location if it's not in PATH (replace with your actual path if needed)
        # 'ffmpeg_location': '/path/to/your/ffmpeg',
    }

    final_fixed_path = None
    original_title_for_button = "downloaded_video"
    actual_extension = "mp4" # Default to merge format

    try:
        with YoutubeDL(ydl_opts) as ydl:
            status_text.info(f"Extracting video info...")
            info_dict = ydl.extract_info(validated_url, download=False)
            original_title_for_button = info_dict.get("title", "youtube_video")
            actual_extension = ydl_opts.get('merge_output_format') or info_dict.get('ext', 'mp4')
            final_fixed_path = os.path.join(YOUTUBE_DOWNLOAD_PATH, f"{fixed_filename_base}.{actual_extension}")
            logger.info(f"Target download path (overwrite): {final_fixed_path}")

            status_text.info(f"Starting download for '{original_title_for_button}'...")
            ydl.download([validated_url])
            # --- Download and merge complete ---

            # === File Verification AFTER download call ===
            time.sleep(1.0) # Increase buffer slightly for merging/filesystem
            if final_fixed_path and os.path.exists(final_fixed_path) and os.path.getsize(final_fixed_path) > 0:
                logger.info(f"Download verified. File exists at fixed path: {final_fixed_path}")
                status_text.success(f"Download ready!") # Simpler message
                progress_bar.empty() # Clear progress bar on success
            else:
                size_info = f"Size: {os.path.getsize(final_fixed_path)}" if os.path.exists(final_fixed_path) else "File not found"
                logger.error(f"Download verification failed. Output file not found or empty at {final_fixed_path}. {size_info}")
                status_text.error(f"Download finished, but the output file is missing or empty.")
                progress_bar.empty()
                final_fixed_path = None # Ensure it's None if verification fails

    except yt_dlp.utils.DownloadError as e:
        error_message = str(e)
        # Extract the core error message if possible
        match = re.search(r"ERROR: (.*?)(?:;|$)", error_message)
        display_error = match.group(1).strip() if match else error_message
        logger.error(f"yt-dlp DownloadError for {validated_url}: {error_message}", exc_info=False)
        status_text.error(f"Download failed: {display_error}")
        progress_bar.empty()
        final_fixed_path = None
    except Exception as e:
        logger.exception(f"Unexpected error during YouTube download {validated_url}: {e}")
        status_text.error(f"An unexpected error occurred: {e}")
        progress_bar.empty()
        final_fixed_path = None

    # --- Provide download button using the fixed path but original title ---
    if final_fixed_path: # Check if verification succeeded
        try:
            safe_user_filename = safe_filename(f"{original_title_for_button}.{actual_extension}")

            with open(final_fixed_path, "rb") as file_data:
                btn_data = file_data.read()

            download_button_placeholder.download_button(
                label=f"Download '{safe_user_filename}'", # Show original title
                data=btn_data,
                file_name=safe_user_filename, # Offer original title for saving
                mime=f"video/{actual_extension}" # Use actual extension
            )
            # status_text.empty() # Clear status after button is shown? Optional.

        except Exception as e:
            st.sidebar.error(f"Error providing download button: {e}")
            logger.error(f"Error reading file or creating download button for {final_fixed_path}: {e}")
