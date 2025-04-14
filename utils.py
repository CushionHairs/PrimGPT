import os
import base64
import re
from io import BytesIO
from urllib.parse import urlparse, parse_qs, quote_plus
import webbrowser
from typing import Optional, Union # Add Union

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError

# Configure logger for utils if needed, or rely on main app logger
import logging
logger = logging.getLogger(__name__)


# --- UI Helpers ---

def set_bg_hack(main_bg: str):
    """Sets the background image for the Streamlit app."""
    # ... (code remains the same) ...
    main_bg_ext = os.path.splitext(main_bg)[1].lstrip('.') # Get extension like 'jpg'
    try:
        with open(main_bg, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{encoded_string});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        logger.warning(f"Background image file not found: {main_bg}")
        st.warning("Could not load background image.")
    except Exception as e:
         logger.error(f"Error setting background hack: {e}")


def sidebar_bg(side_bg: str):
    """Sets the background image for the Streamlit sidebar."""
    # ... (code remains the same) ...
    side_bg_ext = os.path.splitext(side_bg)[1].lstrip('.') # Get extension like 'jpg'
    try:
        with open(side_bg, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
         logger.warning(f"Sidebar background image file not found: {side_bg}")
         st.warning("Could not load sidebar background image.")
    except Exception as e:
         logger.error(f"Error setting sidebar background: {e}")

# --- Image Helpers ---

# --- MODIFIED image_to_base64 ---
def image_to_base64(image_input: Union[Image.Image, bytes], is_bytes: bool = False) -> Optional[str]:
    """
    Converts a PIL Image object OR image bytes to a base64 encoded string.

    Args:
        image_input: Either a PIL Image object or bytes representing an image.
        is_bytes: Set to True if image_input is already bytes.

    Returns:
        Base64 encoded string, or None if conversion fails.
    """
    try:
        if is_bytes:
            # If input is already bytes, just encode it
            if isinstance(image_input, bytes):
                return base64.b64encode(image_input).decode("utf-8")
            else:
                logger.error("Input provided as bytes, but is not type bytes.")
                return None
        else:
            # If input is a PIL Image object
            if isinstance(image_input, Image.Image):
                buffered = BytesIO()
                # Determine format; default to PNG if unknown or problematic
                img_format = image_input.format if image_input.format else "PNG"
                if img_format not in ["PNG", "JPEG", "GIF", "WEBP"]:
                     logger.warning(f"Unsupported image format '{img_format}', saving as PNG.")
                     img_format = "PNG"
                image_input.save(buffered, format=img_format)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                 logger.error("Input provided as Image object, but is not type PIL.Image.Image.")
                 return None
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}", exc_info=True)
        return None
# --- END MODIFIED image_to_base64 ---

def display_meme(image_url: str, caption: str):
    """Downloads, resizes, and displays a meme image with a caption."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'} # Add user agent
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status() # Check for HTTP errors

        # Try to open image, handle potential errors
        try:
             image = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
             logger.error(f"Could not identify image format from URL: {image_url}")
             st.error("Failed to load meme image (invalid format).")
             return
        except Exception as img_err: # Catch other PIL errors
             logger.error(f"Error opening image from URL {image_url}: {img_err}")
             st.error(f"Failed to process meme image: {img_err}")
             return


        # Calculate the new size while maintaining the aspect ratio
        max_size = 512
        width, height = image.size
        if width == 0 or height == 0:
             st.warning("Meme image has zero dimension, cannot resize.")
             return # Cannot resize zero-dimension image

        aspect_ratio = width / height

        if width > height:
            new_width = min(max_size, width) # Don't upscale
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_size, height) # Don't upscale
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are positive integers
        new_width = max(1, int(new_width))
        new_height = max(1, int(new_height))

        resized_image = image.resize((new_width, new_height))
        img_base64 = image_to_base64(resized_image) # Use the updated function

        if not img_base64:
            st.error("Failed to convert resized meme image for display.")
            return

        # Extract subheader text from caption using regex
        # Import MEME_PROMPT_TEMPLATE if defined elsewhere, or define here
        MEME_PROMPT_TEMPLATE = r'Sends a meme of \{.*?\} with the caption \{(?P<caption>.*?)\}' # Added definition
        match = re.search(MEME_PROMPT_TEMPLATE, caption)
        subheader_text = match.group('caption').strip() if match else "Meme Caption"

        # Display in an expander
        with st.expander("😂 Meme Maestro's Creation", expanded=True):
            # Sanitize subheader text for HTML display
            import html
            safe_subheader = html.escape(subheader_text)
            st.markdown(
                f'<h3 style="text-align: center;">{safe_subheader}</h3>',
                unsafe_allow_html=True
            )
            # Center the image using markdown container
            st.markdown(
                f'<div style="display: flex; justify-content: center;">'
                f'<img src="data:image/png;base64,{img_base64}" alt="Generated Meme" style="max-width: 100%; height: auto;">' # Added style
                f'</div>',
                unsafe_allow_html=True
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching meme image URL {image_url}: {e}")
        st.error(f"Error fetching meme image: {e}")
    except Exception as e:
        logger.error(f"Error displaying meme from {image_url}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while displaying the meme: {e}")


# --- URL Helpers ---

def is_valid_video_id(video_id: str) -> bool:
    """Validates YouTube video ID format."""
    # Ensure it's a string before matching
    return isinstance(video_id, str) and re.match(r'^[\w-]{11}$', video_id) is not None

def get_youtube_video_id(video_url: str) -> Optional[str]:
    """Extracts YouTube video ID from various URL formats."""
    if not isinstance(video_url, str):
        logger.debug(f"Input URL is not a string: {type(video_url)}")
        return None
    try:
        parsed_url = urlparse(video_url)
        hostname = parsed_url.hostname.lower() if parsed_url.hostname else ''

        # Handle youtu.be short links
        if hostname == 'youtu.be':
            video_id = parsed_url.path[1:] # Remove leading '/'
            return video_id if is_valid_video_id(video_id) else None

        # Handle standard youtube.com links (www, m, no subdomain)
        if hostname.endswith('youtube.com'):
            # Check /embed/ path first
            if '/embed/' in parsed_url.path:
                video_id = parsed_url.path.split('/embed/')[-1].split('?')[0] # Get part after /embed/, remove query params
                return video_id if is_valid_video_id(video_id) else None
            # Check /watch path and 'v' query parameter
            elif parsed_url.path == '/watch':
                qs = parse_qs(parsed_url.query)
                video_id_list = qs.get('v', [])
                if video_id_list:
                    video_id = video_id_list[0]
                    return video_id if is_valid_video_id(video_id) else None
            # Check /shorts/ path
            elif '/shorts/' in parsed_url.path:
                 video_id = parsed_url.path.split('/shorts/')[-1].split('?')[0]
                 return video_id if is_valid_video_id(video_id) else None

        # If no standard format matches
        logger.debug(f"Could not extract video ID from URL: {video_url}")
        return None
    except Exception as e:
        logger.error(f"Error parsing YouTube URL {video_url}: {e}")
        return None

def validate_youtube_url(video_url: str) -> Optional[str]:
    """Validates and canonicalizes YouTube URL. Shows error in sidebar on failure."""
    video_id = get_youtube_video_id(video_url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    else:
        # Display error only if a URL was actually provided
        if video_url:
             st.sidebar.error(f'Invalid YouTube link: "{video_url[:50]}..."', icon="🚨")
        # else: Don't show error if input was empty
        return None

# --- General Helpers ---
def safe_filename(filename: str, max_len: int = 100) -> str:
    """Removes or replaces characters unsafe for filenames and limits length."""
    if not isinstance(filename, str):
        filename = "invalid_filename"

    # Remove or replace potentially problematic characters
    # Keep alphanumeric, underscores, hyphens, periods. Replace others.
    filename = re.sub(r'[^\w\-.]', '_', filename)

    # Remove leading/trailing underscores/periods/hyphens/spaces
    filename = filename.strip('._- ')

    # Replace multiple consecutive underscores/hyphens with a single one
    filename = re.sub(r'[-_]+', '_', filename)

    # Ensure filename is not empty after cleaning
    if not filename:
        filename = "untitled"

    # Limit length while preserving extension
    if len(filename) > max_len:
        name, ext = os.path.splitext(filename)
        # Ensure extension is not excessively long itself
        ext = ext[:max_len//4] # Limit extension length too
        name = name[:max_len - len(ext) -1] # Truncate name part, leave space for potential underscore if needed
        filename = name.rstrip('._-') + ext # Reassemble, ensure no trailing separators on name

    return filename

def ani_newLink(query: str):
    """
    Generates a Google search URL for the given anime query (watching)
    and opens it in a new browser tab. Handles spaces correctly.
    """
    if not query:
        st.warning("Please enter an animation title.")
        return

    try:
        # Encode the query including spaces (+) and add " 다시보기" (rewatch)
        search_term = f"{query} 다시보기" # "Rewatch" in Korean
        encoded_query = quote_plus(search_term) # Encodes spaces to '+' and other special chars

        # Construct the final URL
        search_url = f"https://www.google.com/search?q={encoded_query}"
        logger.info(f"Generated anime search URL: {search_url}")

        # Display the link in Streamlit (optional, good for feedback)
        st.markdown(f"Searching Google for: [{search_term}]({search_url})", unsafe_allow_html=True)
        st.caption("(Opening in new browser tab...)")

        # Open the URL in a new browser tab
        webbrowser.open_new_tab(search_url)
        st.toast(f"Opened search for '{query}'", icon="✅")

    except Exception as e:
        st.error(f"Could not open search link: {e}")
        logger.error(f"Error opening webbrowser for anime query '{query}': {e}", exc_info=True)
