# app.py
import streamlit as st
import logging
import traceback # For detailed error logging

# Configure basic logging for the entry point
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Ensure local modules can be imported (adjust sys.path if needed, though usually not necessary with simple structures)
    # import sys
    # import os
    # sys.path.append(os.path.dirname(__file__))

    import PrimGPT # Import the main application module
    PrimGPT.main() # Run the main function

except ImportError as ie:
     logger.error(f"Import Error: {ie}. Please ensure all modules are in the correct path and dependencies are installed.", exc_info=True)
     st.error(f"🚨 Critical Error: Could not load application components ({ie}). Please check the logs and ensure all files are present and dependencies (`requirements.txt`) are installed.")
     st.info("Try running `pip install -r requirements.txt`.")
except Exception as e:
    logger.error("An unexpected critical error occurred in the application.", exc_info=True)
    st.error(f"🚨 An unexpected error occurred: {e}")
    # Optionally display traceback for debugging (be cautious in production)
    st.code(traceback.format_exc())

