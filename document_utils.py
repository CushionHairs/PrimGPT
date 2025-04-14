import os
import logging
from typing import Optional

import streamlit as st
import dill as pickle
from pydub import AudioSegment, exceptions as pydub_exceptions # Import exceptions
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader # Use UnstructuredLoader directly
# from langchain.docstore.document import Document # Not directly used here, but implicitly by loaders/splitters

# Local imports
from openai_utils import transcribe_audio # Assuming transcribe_audio is here or imported correctly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DOC_DATA_PATH = "./doc_data/"
os.makedirs(DOC_DATA_PATH, exist_ok=True)

# --- Audio Processing ---

def load_and_convert_audio(input_file_path: str, output_ogg_path: str) -> bool:
    """Loads audio, converts to OGG Opus format for Whisper."""
    try:
        # Ensure input file exists before attempting to load
        if not os.path.exists(input_file_path):
             raise FileNotFoundError(f"Audio input file not found: {input_file_path}")

        file_extension = os.path.splitext(input_file_path)[1].lower()
        file_format = file_extension[1:] if file_extension else None # 'mp3', 'm4a', etc.

        if not file_format:
             logger.error(f"Could not determine file format for: {input_file_path}")
             st.error("Could not determine audio file format.")
             return False

        logger.info(f"Attempting to load audio: {input_file_path} with format {file_format}")
        sound = None

        # Use explicit format where needed, fallback to default detection
        supported_explicit = ['m4a', 'mp4'] # Add others if needed
        if file_format in supported_explicit:
            try:
                sound = AudioSegment.from_file(input_file_path, format=file_format)
            except pydub_exceptions.CouldntDecodeError:
                 logger.warning(f"Failed loading {input_file_path} explicitly as '{file_format}', trying auto-detect.")
                 # Fall through to auto-detect below
            except Exception as e: # Catch other potential load errors
                 logger.error(f"Error loading {input_file_path} as '{file_format}': {e}")
                 # Fall through to auto-detect

        # If explicit load failed or format wasn't special cased, try auto-detect
        if sound is None:
             try:
                  sound = AudioSegment.from_file(input_file_path)
                  logger.info(f"Successfully loaded {input_file_path} using auto-detection.")
             except pydub_exceptions.CouldntDecodeError as decode_err:
                  logger.error(f"Pydub failed to decode audio file {input_file_path}: {decode_err}")
                  st.error(f"Failed to decode audio file: {os.path.basename(input_file_path)}. It might be corrupted or an unsupported format/codec. Ensure ffmpeg is correctly installed.")
                  return False
             except Exception as e:
                  logger.exception(f"Unexpected error loading audio file {input_file_path}: {e}")
                  st.error(f"An unexpected error occurred while loading the audio file: {e}")
                  return False

        logger.info("Audio loaded. Converting to OGG Opus...")
        # Export parameters optimized for Whisper API
        sound.export(
            output_ogg_path,
            format="ogg",
            codec="libopus",
            bitrate="16k", # Slightly higher bitrate for potentially better quality
            parameters=["-ac", "1", "-application", "voip", "-frame_duration", "60"] # Mono, voice hint, longer frame duration common for Opus
        )

        # Verify output file exists and has size
        if os.path.exists(output_ogg_path) and os.path.getsize(output_ogg_path) > 0:
             logger.info(f"Audio converted and saved to: {output_ogg_path}")
             return True
        else:
             logger.error(f"Failed to create or save valid OGG file: {output_ogg_path}")
             st.error("Audio conversion failed (output file missing or empty).")
             return False

    except FileNotFoundError as fnf_error:
        logger.error(str(fnf_error))
        st.error("Audio file seems to be missing.")
        return False
    except pydub_exceptions.CouldntEncodeError as encode_err:
        logger.error(f"Pydub failed to encode audio to OGG for {input_file_path}: {encode_err}")
        st.error(f"Failed to encode audio to the required format. Ensure ffmpeg/libopus is correctly installed: {encode_err}")
        return False
    except Exception as e:
        logger.exception(f"Error during audio load/convert for {input_file_path}: {e}")
        st.error(f"Failed to process audio file: {e}. Ensure ffmpeg is installed and accessible.")
        return False


def save_transcription(text: str, base_output_path: str) -> Optional[str]: # Return Optional
    """Saves transcription text to a .txt file."""
    txt_path = f"{base_output_path}.txt"
    try:
        with open(txt_path, "w", encoding='utf-8') as txt_file:
            txt_file.write(text)
        logger.info(f"Transcription saved to: {txt_path}")
        return txt_path
    except IOError as e:
        logger.error(f"Error saving transcription to {txt_path}: {e}")
        st.error("Could not save the transcription text.")
        return None # Return None on failure


def cleanup_files(*file_paths: str):
    """Safely removes temporary files, ignoring errors."""
    for file_path in file_paths:
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                # Log warning but continue, cleanup is best-effort
                logger.warning(f"Could not remove temporary file {file_path}: {e}")

# --- Document & Vector Store Processing ---

def create_vector_db_from_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> Optional[bytes]:
    """Loads text from file, splits, creates, and serializes FAISS DB."""
    try:
        # Ensure embeddings can be created (requires API key)
        embeddings = OpenAIEmbeddings()
    except Exception as e:
         logger.error(f"Failed to initialize OpenAIEmbeddings: {e}")
         st.error("Failed to initialize embeddings. Check API key and connectivity. Cannot create vector DB.")
         return None

    try:
        logger.info(f"Loading document: {file_path}")
        # Using UnstructuredLoader which handles various text-based formats
        # Specify encoding if known, otherwise let Unstructured try common ones
        loader = UnstructuredLoader(file_path, encoding="utf-8", errors="ignore") # Try utf-8 first
        doc_contents = loader.load() # Returns List[Document]

        if not doc_contents:
            logger.warning(f"No content loaded from file: {file_path}. Trying default encoding.")
            # Try again without specific encoding
            try:
                 loader = UnstructuredLoader(file_path)
                 doc_contents = loader.load()
                 if not doc_contents:
                      logger.warning(f"Still no content loaded from file: {file_path} after trying default encoding.")
                      st.warning("Could not extract any text content from the document.")
                      return None
            except Exception as load_err_default:
                 logger.error(f"Failed to load document {file_path} even with default encoding: {load_err_default}")
                 st.error(f"Failed to load document content: {load_err_default}")
                 return None


        logger.info("Splitting document content...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, # Use standard len
            is_separator_regex=False, # Use default separators
        )
        docs = text_splitter.split_documents(doc_contents)

        if not docs:
            logger.warning(f"Document splitting resulted in no chunks for: {file_path}")
            st.warning("Failed to split the document into processable chunks.")
            return None

        logger.info(f"Document split into {len(docs)} chunks. Creating FAISS DB...")
        doc_db = FAISS.from_documents(docs, embeddings)
        serialized_db = doc_db.serialize_to_bytes() # Serializes the FAISS index
        logger.info(f"FAISS DB created and serialized for: {file_path}")
        return serialized_db

    except FileNotFoundError:
        logger.error(f"Document file not found for DB creation: {file_path}")
        st.error("The specified document file could not be found.")
        return None
    except Exception as e:
        # Catch specific Unstructured errors if possible
        logger.exception(f"Error creating vector DB from {file_path}: {e}")
        st.error(f"Failed to process the document into a database: {e}")
        return None

def save_vector_db(serialized_db: bytes, base_output_path: str) -> Optional[str]: # Return Optional
    """Saves serialized vector DB to a .bin file."""
    bin_path = f"{base_output_path}.bin"
    try:
        with open(bin_path, "wb") as file:
            pickle.dump(serialized_db, file)
        logger.info(f"Vector DB saved to: {bin_path}")
        return bin_path
    except IOError as e:
        logger.error(f"Error saving vector DB to {bin_path}: {e}")
        st.error("Could not save the document database.")
        return None # Return None on failure


# --- Main Pipeline ---

def file_processing_pipeline(uploaded_file_path: str, original_filename: str) -> bool:
    """
    Processes uploaded files (audio or documents) into searchable vector DBs.
    Handles temporary file cleanup. Returns True on success, False on failure.
    Displays status messages in the sidebar.
    """
    from utils import safe_filename
    
    base_filename = os.path.splitext(original_filename)[0]
    file_extension = os.path.splitext(original_filename)[1].lower()
    # Ensure base filename is safe for filesystem operations
    safe_base_filename = safe_filename(base_filename, max_len=80) # Use safe name for outputs
    output_base_path = os.path.join(DOC_DATA_PATH, safe_base_filename)

    # Files to cleanup at the end
    files_to_remove = [uploaded_file_path] # Start with the uploaded temp file
    final_bin_path = f"{output_base_path}.bin" # Expected final output

    st.sidebar.info(f"Processing '{original_filename}'...")

    try:
        # --- Audio File Handling ---
        if file_extension in ['.mp3', '.m4a', '.wav', '.ogg', '.flac']:
            ogg_path = f"{output_base_path}_converted.ogg" # More specific temp name
            txt_path = None # Initialize
            files_to_remove.append(ogg_path) # Add ogg to cleanup list

            st.sidebar.text("Converting audio...")
            if not load_and_convert_audio(uploaded_file_path, ogg_path):
                # Error shown by load_and_convert_audio
                return False # Stop processing

            st.sidebar.text("Transcribing audio (Whisper)...")
            transcription_text = transcribe_audio(ogg_path)
            if transcription_text is None:
                # Error shown by transcribe_audio
                return False

            st.sidebar.text("Saving transcription...")
            txt_path = save_transcription(transcription_text, output_base_path)
            if txt_path is None: # Check for failure
                 # Error shown by save_transcription
                 return False
            files_to_remove.append(txt_path) # Add txt to cleanup list (even if DB fails)

            st.sidebar.text("Creating document database...")
            serialized_db = create_vector_db_from_file(txt_path)
            if serialized_db is None:
                # Error shown by create_vector_db_from_file
                return False

            st.sidebar.text("Saving database...")
            saved_bin_path = save_vector_db(serialized_db, output_base_path)
            if saved_bin_path is None:
                # Error shown by save_vector_db
                return False

            # Success! final_bin_path should match saved_bin_path
            st.sidebar.success(f"'{original_filename}' processed successfully!")
            return True

        # --- Document File Handling ---
        elif file_extension in ['.pdf', '.txt', '.docx', '.md', '.html', '.csv']:
            st.sidebar.text("Creating document database...")
            # Use the original uploaded file path directly for text documents
            serialized_db = create_vector_db_from_file(uploaded_file_path)
            if serialized_db is None:
                # Error shown by create_vector_db_from_file
                return False

            st.sidebar.text("Saving database...")
            saved_bin_path = save_vector_db(serialized_db, output_base_path)
            if saved_bin_path is None:
                 # Error shown by save_vector_db
                 return False

            # Success!
            st.sidebar.success(f"'{original_filename}' processed successfully!")
            return True

        # --- Unsupported File Type ---
        else:
            logger.warning(f"Unsupported file type uploaded: {original_filename}")
            st.sidebar.error(f"Unsupported file type: '{file_extension}'. Cannot process.")
            return False # Return False for unsupported types

    except Exception as e:
        # Catch any unexpected errors during the pipeline flow
        logger.exception(f"Error in processing pipeline for {original_filename}: {e}")
        st.sidebar.error(f"An unexpected error occurred during processing: {e}")
        return False # Return False on unexpected exception

    finally:
        # --- Cleanup ---
        # Remove all intermediate files EXCEPT the final .bin file if processing was successful
        # Determine if success based on existence of the final .bin file
        was_successful = os.path.exists(final_bin_path)

        files_to_really_remove = []
        for f_path in files_to_remove:
             if was_successful and f_path == final_bin_path:
                  continue # Don't remove the final product on success
             files_to_really_remove.append(f_path)

        if files_to_really_remove:
             logger.info(f"Cleaning up files for '{original_filename}': {files_to_really_remove}")
             cleanup_files(*files_to_really_remove)