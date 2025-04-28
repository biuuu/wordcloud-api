#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import logging
import re
from collections import Counter
import math # For log in IDF calculation
from flask import Flask, request, jsonify, send_from_directory
from wordcloud import WordCloud
import numpy as np
from PIL import Image

# --- Text Processing Imports ---
try:
    import jieba
    import jieba.analyse # Import analyse submodule
    JIEBA_AVAILABLE = True
    logging.info("Jieba library and analyse submodule found.")
except ImportError:
    JIEBA_AVAILABLE = False
    # Log a critical error if Jieba is missing, as it's essential now
    logging.critical("Jieba library or analyse submodule not found. Service cannot function.", exc_info=True)
    # Optionally exit if the service is unusable without Jieba
    # import sys
    # sys.exit("Exiting: Jieba library is mandatory and was not found.")
# --- End Text Processing Imports ---


# ================== CONFIGURATION ==================
# Determine the directory where the script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save generated images (relative to the script directory)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_images')

# --- !!! CRITICAL CONFIGURATION !!! ---
# Default font path. MUST point to a valid font file that supports Chinese characters.
# Replace '/path/to/your/chinese_font.ttf' with the actual path on your server.
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf'
# --- !!! END CRITICAL CONFIGURATION !!! ---

# Filename for the Chinese stopwords file (expected in the same directory as the script)
STOPWORDS_FILENAME = 'baidu_stopwords.txt' # Or your preferred filename
STOPWORDS_FILE_PATH = os.path.join(SCRIPT_DIR, STOPWORDS_FILENAME)

# Basic logging setup (logs to console)
# Use INFO level for general messages, DEBUG for more detailed steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e:
        logging.error(f"Failed to create output directory {OUTPUT_DIR}: {e}", exc_info=True)
        # Exit if cannot create output directory
        import sys
        sys.exit("Exiting: Failed to create image output directory.")
# ================== END CONFIGURATION ==================

# --- Load Chinese Stopwords At Startup ---
LOADED_CHINESE_STOPWORDS = set()
# Keep a minimal fallback list in case the file is missing or empty
DEFAULT_HARDCODED_CHINESE_STOPWORDS = set(['的', '了', '和', '是', '就', '都', '而', '及'])
try:
    logging.info(f"Attempting to load Chinese stopwords from: {STOPWORDS_FILE_PATH}")
    with open(STOPWORDS_FILE_PATH, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace, add non-empty lines to the set
        LOADED_CHINESE_STOPWORDS = {line.strip() for line in f if line.strip()}
    if LOADED_CHINESE_STOPWORDS:
        logging.info(f"Successfully loaded {len(LOADED_CHINESE_STOPWORDS)} Chinese stopwords from {STOPWORDS_FILENAME}.")
    else:
        # If file is empty, use fallback
        logging.warning(f"Stopword file '{STOPWORDS_FILENAME}' was found but contained no words. Falling back to default list.")
        LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except FileNotFoundError:
    # If file not found, use fallback
    logging.warning(f"Stopword file '{STOPWORDS_FILENAME}' not found at {STOPWORDS_FILE_PATH}. Falling back to default list.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except Exception as e:
    # On other errors, log the error and use fallback
    logging.error(f"An error occurred while loading stopwords from {STOPWORDS_FILENAME}: {e}", exc_info=True)
    logging.warning("Falling back to default hardcoded Chinese stopwords due to loading error.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
finally:
     # Log the final count of stopwords being used
     logging.info(f"Service will use {len(LOADED_CHINESE_STOPWORDS)} Chinese stopwords for filtering.")
# --- End Stopword Loading ---


# Initialize Flask application
app = Flask(__name__)


# --- Helper Function for Text Cleaning ---
def clean_chinese_text(text):
    """Basic cleaning: Keep Chinese characters and spaces."""
    if not isinstance(text, str): return ""
    # Keep Chinese Unicode range (\u4e00-\u9fff) and whitespace (\s)
    cleaned_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    # Normalize multiple spaces/newlines to single spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# --- Text Processing Function (Message TF-IDF) ---
def calculate_message_tfidf(messages, combined_stopwords, top_k=200):
    """
    Calculates TF-IDF scores based on individual messages as documents.
    Assumes user dictionary words are already added to jieba globally for the request.
    Returns a dictionary {word: scaled_tfidf_score}.
    """
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required for TF-IDF.")
    if not messages:
        logging.warning("calculate_message_tfidf received an empty message list.")
        return {}

    total_messages = len(messages)
    doc_freq = Counter()        # Counts in how many messages a word appears
    term_freq_total = Counter() # Counts total occurrences of a word across all messages

    logging.debug(f"Processing {total_messages} messages for TF-IDF...")

    # Pass 1: Tokenize, count TF per message, update DF and Total TF
    for i, msg in enumerate(messages):
        cleaned_msg = clean_chinese_text(msg)
        if not cleaned_msg: continue # Skip empty messages

        # Tokenize using jieba (respects globally added user words)
        tokens = jieba.cut(cleaned_msg)

        # Filter tokens using the combined stopword list and length check
        filtered_tokens = []
        for token in tokens:
            word = token.strip() # Strip whitespace first
            # Apply filters: non-empty, not stopword, length > 1
            if word and word not in combined_stopwords and len(word) > 1:
                filtered_tokens.append(word)

        if filtered_tokens:
            # Update total term frequency
            term_freq_total.update(filtered_tokens)
            # Update document frequency (count unique words per message)
            doc_freq.update(set(filtered_tokens))
        # else:
        #     logging.debug(f"Message {i+1} yielded no tokens after cleaning/filtering.")


    # Check if any terms were found at all
    if not term_freq_total:
        logging.warning("No valid terms found across all messages after tokenization and filtering for TF-IDF.")
        return {}

    # Pass 2: Calculate TF-IDF score for each unique term
    tfidf_scores = {}
    num_terms = len(term_freq_total)
    logging.debug(f"Calculating TF-IDF for {num_terms} unique terms...")

    for term, total_tf in term_freq_total.items():
        df = doc_freq.get(term, 0)
        if df == 0: continue # Safeguard, should not happen

        # Calculate IDF (Inverse Document Frequency) with smoothing
        # Add 1 to numerator and denominator to prevent division by zero and handle terms appearing in all docs
        idf = math.log((total_messages + 1) / (df + 1)) + 1.0 # Add 1.0 ensures IDF is always >= 1

        # TF-IDF score = Term Frequency * Inverse Document Frequency
        tfidf = total_tf * idf
        tfidf_scores[term] = tfidf

    # Sort terms by TF-IDF score (highest first)
    # Use items() for Python 3 compatibility
    sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

    # Select top K terms based on the request
    top_terms = sorted_terms[:top_k]

    if not top_terms:
        logging.warning(f"No terms remained after TF-IDF calculation and selecting top {top_k}.")
        return {}

    # Scale the scores to be suitable for WordCloud visual representation
    # Option 1: Scale relative to the max score
    max_score = top_terms[0][1] if top_terms else 1.0
    scale_factor = 1000 / max_score if max_score > 0 else 1000 # Scale max to 1000
    # Option 2: Use a fixed large multiplier (simpler, might need tuning)
    # scale_factor = 500 # Example fixed factor

    final_frequencies = {
        word: max(1, int(score * scale_factor)) # Ensure weight is at least 1
        for word, score in top_terms
    }

    logging.info(f"TF-IDF calculation complete. Returning {len(final_frequencies)} terms (top {top_k} requested).")
    return final_frequencies

# --- Text Processing Function (Global Frequency Count) ---
def process_text_chinese_frequency(text, combined_stopwords):
    """
    Processes Chinese text using Jieba frequency count method.
    Assumes user dictionary words already added. Filters using combined_stopwords.
    """
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required.")

    cleaned_text = clean_chinese_text(text)
    if not cleaned_text:
        logging.warning("Frequency method received empty text after cleaning.")
        return Counter()

    tokens = jieba.cut(cleaned_text)

    # Filter tokens based on stopwords and length
    words = []
    for token in tokens:
        word = token.strip() # Strip first
        # Apply filters
        if word and word not in combined_stopwords and len(word) > 1:
            words.append(word)

    if not words: logging.warning(f"Frequency method yielded no words after processing and filtering.")
    # Return a frequency count of the valid words
    return Counter(words)
# --- End Text Processing Functions ---


# --- Mask Generation Functions ---
def create_circle_mask(width, height):
    """Creates a NumPy array for a circular mask."""
    if width <= 0 or height <= 0: raise ValueError("Mask dimensions must be positive.")
    radius = min(width, height) // 2; center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]; mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image

def create_ellipse_mask(width, height):
    """Creates a NumPy array for an elliptical mask."""
    if width <= 0 or height <= 0: raise ValueError("Mask dimensions must be positive.")
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = max(1, width // 2), max(1, height // 2)
    y, x = np.ogrid[:height, :width]; mask_boolean = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2) <= 1
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image
# --- End Mask Generation Functions ---


# --- Flask Routes ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    """
    Generates Chinese word cloud based on request parameters.
    Supports 'message_tfidf' (default) and 'global_frequency' weighting.
    """
    # Check prerequisite at the start of the request
    if not JIEBA_AVAILABLE:
        logging.error("Jieba not available, cannot process request.")
        return jsonify({"error": "Server configuration error: Text processing library (Jieba) is unavailable."}), 503 # Service Unavailable

    # Ensure request is JSON
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    # Log received keys for debugging without exposing sensitive text
    logging.info(f"Received request data (keys): {list(data.keys())}")

    # --- Extract Data and Apply Defaults ---
    raw_text = data.get('text')
    custom_stopwords = data.get('custom_stopwords', [])
    user_dict_words = data.get('user_dict_words', [])
    # Weighting scheme determines how word importance is calculated
    weighting_scheme = data.get('weighting_scheme', 'message_tfidf').lower()

    # Visual options
    shape = data.get('shape', 'rectangle').lower()
    try:
        width = int(data.get('width', 800)); height = int(data.get('height', 600))
        max_words = int(data.get('max_words', 200)) # General limit for words displayed
        scale = float(data.get('scale', 1))         # Resolution multiplier
        top_k = int(data.get('top_k', max_words))   # How many top words to consider from TF-IDF
    except (ValueError, TypeError): return jsonify({"error": "width, height, max_words, top_k must be integers; scale must be a number."}), 400
    background_color = data.get('background_color', None) # None allows transparency with RGBA mode
    font_path_req = data.get('font_path', None)
    mode = data.get('mode', 'RGBA') # Use RGBA for transparency support
    colormap_name = data.get('colormap', 'tab20') # Default color scheme

    # --- Input Validation ---
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip(): return jsonify({"error": "Missing or invalid 'text' field (must be a non-empty string)"}), 400
    if width <= 0 or height <= 0 or max_words <= 0 or scale <= 0 or top_k <= 0: return jsonify({"error": "width, height, max_words, scale, top_k must be positive numbers."}), 400
    if weighting_scheme not in ['global_frequency', 'message_tfidf']: return jsonify({"error": "Invalid weighting_scheme. Use 'global_frequency' or 'message_tfidf'."}), 400
    if not isinstance(custom_stopwords, list): return jsonify({"error": "custom_stopwords must be a list of strings."}), 400
    if not isinstance(user_dict_words, list): return jsonify({"error": "user_dict_words must be a list of strings."}), 400

    # --- Font Path Validation ---
    final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
    if not os.path.exists(final_font_path):
        logging.error(f"Font file not found at selected path: {final_font_path}")
        if font_path_req: return jsonify({"error": f"Requested font file '{font_path_req}' not found on server."}), 400
        else: return jsonify({"error": f"Default font file ('{DEFAULT_FONT_PATH}') not found or not configured correctly on the server."}), 500

    # --- Initialize variables for results ---
    frequencies = {}
    actual_word_count = 0
    # Track the method actually used (useful if there are fallbacks, though none currently)
    weighting_scheme_used = weighting_scheme

    try:
        # --- Apply User Dictionary (Affects BOTH weighting schemes) ---
        if user_dict_words:
            added_count = 0
            for word in user_dict_words:
                # Add word to Jieba's dictionary for the current process lifespan
                if isinstance(word, str) and word.strip():
                    jieba.add_word(word.strip())
                    added_count += 1
            if added_count > 0:
                logging.info(f"Temporarily added {added_count} custom words to Jieba dictionary for this request.")

        # --- Prepare Combined Stopwords (Used for filtering in both schemes) ---
        combined_stopwords = LOADED_CHINESE_STOPWORDS.copy() # Start with base list
        if custom_stopwords:
            # Add valid strings from the request's custom list
            combined_stopwords.update(sw for sw in custom_stopwords if isinstance(sw, str) and sw.strip())
        logging.debug(f"Total combined stopwords count for filtering: {len(combined_stopwords)}")

        # --- Calculate Word Frequencies/Weights based on chosen scheme ---
        logging.info(f"Calculating word weights using scheme: '{weighting_scheme}'...")

        if weighting_scheme == 'message_tfidf':
            # Split text into messages (documents) based on newlines
            messages = [msg for msg in raw_text.splitlines() if msg.strip()]
            if not messages:
                 logging.warning("Input text contains no processable message lines after splitting.")
                 return jsonify({"error": "Input text contains no valid message lines."}), 400

            logging.info(f"Split input into {len(messages)} messages for TF-IDF calculation.")
            # Calculate scaled TF-IDF scores
            frequencies = calculate_message_tfidf(messages, combined_stopwords, top_k)
            weighting_scheme_used = 'message_tfidf' # Confirm method used

        else: # Default to 'global_frequency'
            # Calculate global frequency count across the entire text
            word_counts = process_text_chinese_frequency(raw_text, combined_stopwords)
            if not word_counts:
                 # The function logs a warning, return error response
                 return jsonify({"error": "No processable words found after frequency counting and filtering."}), 400
            # Convert Counter to dict
            frequencies = dict(word_counts)
            weighting_scheme_used = 'global_frequency'

        # Check if the chosen method yielded any results
        if not frequencies:
             logging.error(f"No frequencies generated using method '{weighting_scheme_used}'.")
             return jsonify({"error": f"Failed to extract any valid words using method '{weighting_scheme_used}'. Check input text and stopwords."}), 500 # Internal error seems appropriate if processing succeeds but yields nothing
        actual_word_count = len(frequencies) # How many words before WordCloud limits
        logging.info(f"Text processing complete. Found {actual_word_count} candidate words using '{weighting_scheme_used}'.")


    except RuntimeError as e:
         # Catch specific errors like Jieba missing if startup check failed
         logging.error(f"Runtime error during text processing: {e}", exc_info=True)
         return jsonify({"error": f"Text processing runtime error: {e}"}), 500
    except Exception as e:
        # Catch any other unexpected errors during text processing
        logging.error(f"Unexpected error during text processing: {e}", exc_info=True)
        return jsonify({"error": f"Text processing failed unexpectedly: {e}"}), 500

    # --- Create Shape Mask (if requested) ---
    mask_array = None
    if shape != 'rectangle':
        try:
            if shape == 'circle': mask_array = create_circle_mask(width, height)
            elif shape == 'ellipse': mask_array = create_ellipse_mask(width, height)
            else: logging.warning(f"Unknown shape '{shape}' requested, defaulting to rectangle."); shape = 'rectangle'
        except ValueError as e: return jsonify({"error": f"Invalid dimensions for mask shape '{shape}': {e}"}), 400
        except Exception as e: logging.error(f"Error creating mask: {e}", exc_info=True); return jsonify({"error": "Failed to create shape mask."}), 500

    # --- Generate Word Cloud Image ---
    try:
        logging.info(f"Generating word cloud image (Scheme: {weighting_scheme_used}, Shape: {shape}, MaxWords: {max_words}, Colormap: {colormap_name})...")

        # Instantiate WordCloud object with all configurations
        wordcloud_instance = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            font_path=final_font_path, # Validated font path
            max_words=max_words,       # Let WordCloud limit the final number of words
            mask=mask_array,           # Apply mask if created
            mode=mode,                 # Typically 'RGBA' for transparency
            scale=scale,               # Image resolution scale factor
            colormap=colormap_name,    # Color scheme for words
            # Consider adding include_numbers=False if numbers sometimes slip through
            # include_numbers=False,
            # Stopwords are applied during text processing, not needed here
            # stopwords=None,
        )

        # Generate the word positions and image from the calculated frequencies/weights
        wordcloud_instance.generate_from_frequencies(frequencies)

        # --- Save Image ---
        # Generate a unique filename using UUID to avoid collisions
        filename = f"wc_{uuid.uuid4()}.png" # Use PNG for transparency support
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Save the generated word cloud to the file
        wordcloud_instance.to_file(output_path)
        logging.info(f"Word cloud image successfully saved to: {output_path}")

        # --- Prepare and Return Success Response ---
        # Construct the publicly accessible URL for the image
        image_url = f"{request.host_url}images/{filename}"

        # Get the actual number of words placed by the library
        final_word_count = len(wordcloud_instance.words_)

        return jsonify({
            "imageUrl": image_url,
            "wordCount": final_word_count, # Actual words placed
            "shape": shape,
            "dimensions": {"width": width, "height": height},
            "colormap_used": colormap_name,
            "weighting_scheme_used": weighting_scheme_used # Confirm method used
        }), 200 # OK status

    except FileNotFoundError as e:
         # Error if font file becomes unavailable between check and use
         logging.error(f"Font file error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Internal Error: Font file '{final_font_path}' became unavailable or is invalid."}), 500
    except ValueError as e:
         # Errors from wordcloud library, e.g., if frequencies somehow became invalid
         logging.error(f"Value error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Word cloud generation error: {e}"}), 500
    except Exception as e:
        # Catch-all for other unexpected errors during generation or saving
        logging.error(f"Unexpected error generating or saving word cloud: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate or save word cloud image: {e}"}), 500


@app.route('/images/<path:filename>')
def get_image(filename):
    """Serves the generated image files from the OUTPUT_DIR."""
    # Security check: Ensure the requested path is within the intended directory
    safe_dir = os.path.abspath(OUTPUT_DIR)
    requested_path = os.path.abspath(os.path.join(safe_dir, filename))

    # Prevent accessing files outside the designated output directory
    if not requested_path.startswith(safe_dir):
        logging.warning(f"Attempted directory traversal detected for filename: {filename}")
        return "Forbidden", 403 # Standard "Forbidden" response

    # Use Flask's send_from_directory for robust and safer file serving
    try:
        logging.info(f"Serving image file: {filename}")
        # as_attachment=False attempts to display the image inline in the browser
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        # If the file doesn't exist within the directory
        logging.warning(f"Requested image file not found: {filename}")
        return "Image not found", 404 # Standard "Not Found" response
    except Exception as e:
        # Catch other potential I/O or Flask errors during file serving
        logging.error(f"Error serving image file {filename}: {e}", exc_info=True)
        return "Internal server error", 500

@app.route('/health')
def health_check():
    """Provides a basic health check endpoint for monitoring."""
    # Check essential components like Jieba availability and default font existence
    font_ok = os.path.exists(DEFAULT_FONT_PATH)
    return jsonify({
        "status": "ok",
        "dependencies": { "jieba": "available" if JIEBA_AVAILABLE else "missing" },
        "default_font_found": font_ok
        }), 200

# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize Jieba only if it was successfully imported
    if JIEBA_AVAILABLE:
        try:
            # Initialize Jieba (loads default dictionary, etc.)
            jieba.initialize()
            logging.info("Jieba initialized successfully.")
            # Note: We are NOT setting global TF-IDF stopwords here because we filter manually later
            # to accommodate request-specific custom_stopwords alongside the base list.
        except Exception as e:
            # Log error but allow server to start; route handler will return 503 if Jieba needed but failed init.
            logging.error(f"Failed to initialize Jieba (non-critical at startup): {e}")

    # Perform a critical check for the default font path at startup
    if not os.path.exists(DEFAULT_FONT_PATH):
         # Log a prominent error message if the default font is missing
         logging.error("="*60)
         logging.error(f"CRITICAL STARTUP ERROR: Default font path not found!")
         logging.error(f"Path configured: {DEFAULT_FONT_PATH}")
         logging.error("Requests may fail if they do not provide a valid 'font_path'.")
         logging.error("Please ensure the DEFAULT_FONT_PATH in the script is correct.")
         logging.error("="*60)
         # Consider exiting if the default font is absolutely mandatory for basic operation
         # import sys
         # sys.exit("Exiting: Default font file is essential and was not found.")

    # Start the Flask development server.
    # For production, replace this with a WSGI server like Gunicorn or uWSGI.
    logging.info(f"Starting Flask application server on host 0.0.0.0, port 5000...")
    # Set debug=False for performance and security in production/staging environments
    # Host '0.0.0.0' allows connections from other machines/containers on the network
    app.run(host='0.0.0.0', port=5000, debug=False)