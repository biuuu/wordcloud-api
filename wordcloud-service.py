#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import logging
import re
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from wordcloud import WordCloud
import numpy as np # For mask generation
from PIL import Image # Potentially useful for advanced mask loading/manipulation

# --- Text Processing Imports ---
try:
    import jieba
    JIEBA_AVAILABLE = True
    logging.info("Jieba library found, Chinese/Mixed tokenization enabled.")
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("Jieba library not found. Chinese/Mixed processing ('language=zh') will fail.")

try:
    import nltk
    # Attempt to find necessary data; provides a more informative startup message if missing.
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Pre-load English stopwords for efficiency
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    logging.info("NLTK library and data found, English tokenization/stopwords enabled.")
except (ImportError, LookupError) as e:
    NLTK_AVAILABLE = False
    # Define a basic English stopword list if NLTK is unavailable
    ENGLISH_STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    logging.warning(f"NLTK library/data not available or configured correctly ({e}). Using basic English stopword list for 'language=en'.")
# --- End Text Processing Imports ---


# ================== CONFIGURATION ==================
# Determine the directory where the script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save generated images (relative to the script directory)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_images')

# --- !!! CRITICAL CONFIGURATION !!! ---
# Default font path. MUST point to a valid font file that supports
# the characters you intend to display (e.g., Chinese AND English).
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf' # <--- !!! CHANGE THIS !!!
# --- !!! END CRITICAL CONFIGURATION !!! ---

# Filename for the Chinese stopwords file (expected in the same directory as the script)
STOPWORDS_FILENAME = 'baidu_stopwords.txt'
STOPWORDS_FILE_PATH = os.path.join(SCRIPT_DIR, STOPWORDS_FILENAME)

# Basic logging setup (logs to console)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e:
        logging.error(f"Failed to create output directory {OUTPUT_DIR}: {e}", exc_info=True)
        # Consider exiting if image saving is critical
        # import sys
        # sys.exit(1)
# ================== END CONFIGURATION ==================


# --- Load Stopwords At Startup ---
LOADED_CHINESE_STOPWORDS = set()
# Keep a minimal fallback list in case the file is missing or empty
DEFAULT_HARDCODED_CHINESE_STOPWORDS = set([
    '的', '了', '和', '是', '就', '都', '而', '及', '與', '或', '个', '也', '这', '那'
])

try:
    # Attempt to load stopwords from the specified file using UTF-8 encoding
    logging.info(f"Attempting to load Chinese stopwords from: {STOPWORDS_FILE_PATH}")
    with open(STOPWORDS_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word: # Ignore empty lines
                LOADED_CHINESE_STOPWORDS.add(word)

    if LOADED_CHINESE_STOPWORDS:
        logging.info(f"Successfully loaded {len(LOADED_CHINESE_STOPWORDS)} Chinese stopwords from {STOPWORDS_FILENAME}.")
    else:
        logging.warning(f"Stopword file '{STOPWORDS_FILENAME}' was found but contained no words. Falling back to default list.")
        LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
        logging.info(f"Using default hardcoded Chinese stopwords ({len(LOADED_CHINESE_STOPWORDS)} words).")

except FileNotFoundError:
    logging.warning(f"Stopword file '{STOPWORDS_FILENAME}' not found at {STOPWORDS_FILE_PATH}. Falling back to default list.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
    logging.info(f"Using default hardcoded Chinese stopwords ({len(LOADED_CHINESE_STOPWORDS)} words).")
except Exception as e:
    logging.error(f"An error occurred while loading stopwords from {STOPWORDS_FILENAME}: {e}", exc_info=True)
    logging.warning("Falling back to default hardcoded Chinese stopwords due to loading error.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
    logging.info(f"Using default hardcoded Chinese stopwords ({len(LOADED_CHINESE_STOPWORDS)} words).")
# --- End Stopword Loading ---


# Initialize Flask application
app = Flask(__name__)


# --- Text Processing Functions ---

def process_text_nltk(text, custom_stopwords=None):
    """Process primarily English text using NLTK (if available) or basic split."""
    if not NLTK_AVAILABLE:
        logging.warning("NLTK unavailable, falling back to basic split for tokenization.")
        text = text.lower()
        # Keep alphanumeric and spaces, remove others
        text = re.sub(r'[^\w\s]', '', text)
        # Remove digits
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords:
            stop_words.update(sw.lower() for sw in custom_stopwords)
        words = [word for word in tokens if word and word not in stop_words and len(word) > 1]
    else:
        text = text.lower()
        # NLTK tokenization handles punctuation better
        tokens = nltk.word_tokenize(text)
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords:
            stop_words.update(sw.lower() for sw in custom_stopwords)
        # Keep only alphabetic tokens, remove stopwords and short words
        words = [
            word for word in tokens
            if word.isalpha()
            and word not in stop_words
            and len(word) > 1
        ]
    return Counter(words)

def process_text_mixed_chinese_english(text, custom_stopwords=None):
    """
    Processes text containing both Chinese and English using Jieba.
    Uses combined stopwords (loaded Chinese + NLTK English + custom).
    """
    if not JIEBA_AVAILABLE:
        # This should ideally be caught earlier, but double-check
        raise RuntimeError("Jieba library is required for Chinese/Mixed processing but not installed.")

    # 1. Cleaning: Lowercase English, keep relevant characters, remove digits, normalize spaces.
    text = text.lower()
    # Keep Chinese (\u4e00-\u9fff), English (a-z), and spaces (\s). Remove others.
    cleaned_text = re.sub(r'[^\u4e00-\u9fffA-Za-z\s]', '', text)
    # Remove digits explicitly
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 2. Define combined stopwords set for this request
    combined_stopwords = ENGLISH_STOPWORDS.copy()
    # Add the globally loaded/default Chinese stopwords
    combined_stopwords.update(LOADED_CHINESE_STOPWORDS)
    # Add any custom stopwords provided in the API request
    if custom_stopwords:
        custom_lower = {sw.lower() for sw in custom_stopwords}
        combined_stopwords.update(custom_lower)
    logging.debug(f"Total combined stopwords count for this request: {len(combined_stopwords)}")

    # 3. Tokenization using Jieba (precise mode)
    tokens = jieba.cut(cleaned_text)

    # 4. Filter tokens based on criteria
    words = []
    for token in tokens:
        word = token.strip()
        # Filter out: empty strings, stopwords, single characters, pure numbers
        if word and word not in combined_stopwords and len(word) > 1 and not word.isdigit():
            words.append(word)

    if not words:
         # Log if no words remain after filtering
         logging.warning(f"Text after cleaning and tokenization yielded no words. Original text snippet: {text[:100]}...")

    # Return a frequency count of the valid words
    return Counter(words)

# --- End Text Processing Functions ---


# --- Mask Generation Functions ---
def create_circle_mask(width, height):
    """Creates a NumPy array for a circular mask (black circle on white background)."""
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    radius = min(width, height) // 2
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask_image = np.full((height, width), 255, dtype=np.uint8) # White background (block words)
    mask_image[mask_boolean] = 0 # Black inside circle (allow words)
    return mask_image

def create_ellipse_mask(width, height):
    """Creates a NumPy array for an elliptical mask (black ellipse on white background)."""
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    center_x, center_y = width // 2, height // 2
    # Use half width/height as semi-axes for a standard ellipse filling the space
    radius_x, radius_y = max(1, width // 2), max(1, height // 2) # Ensure radii are at least 1
    y, x = np.ogrid[:height, :width]
    mask_boolean = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2) <= 1
    mask_image = np.full((height, width), 255, dtype=np.uint8) # White background
    mask_image[mask_boolean] = 0 # Black inside ellipse
    return mask_image
# --- End Mask Generation Functions ---


# --- Flask Routes ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    """
    Main endpoint to generate a word cloud.
    Expects JSON payload with 'text' and optional configurations.
    """
    # Ensure request is JSON
    if not request.is_json:
        logging.warning("Received non-JSON request.")
        return jsonify({"error": "Request must be JSON"}), 400

    # Get data from JSON payload
    data = request.get_json()
    logging.info(f"Received request data (keys): {list(data.keys())}") # Log keys, not full data for brevity/privacy

    # --- Extract Data and Apply Defaults ---
    raw_text = data.get('text')
    # Default to 'zh' (mixed Chinese/English) processing if not specified
    language = data.get('language', 'zh').lower()
    # Get custom stopwords list from request, default to empty list
    custom_stopwords = data.get('custom_stopwords', [])

    # WordCloud visual options with defaults
    shape = data.get('shape', 'rectangle').lower()
    try:
        width = int(data.get('width', 800))
        height = int(data.get('height', 600))
        max_words = int(data.get('max_words', 200))
        scale = float(data.get('scale', 1))
    except (ValueError, TypeError):
        return jsonify({"error": "width, height, max_words must be integers; scale must be a number."}), 400

    # Background color: None is often good for transparency with masks
    background_color = data.get('background_color', None)
    # Font path from request (optional override)
    font_path_req = data.get('font_path', None)
    # Mode: RGBA recommended for masks/transparency
    mode = data.get('mode', 'RGBA')
    # Get colormap from request, default to 'tab20'
    colormap_name = data.get('colormap', 'tab20')
    logging.info(f"Using colormap: {colormap_name}") # Log which colormap is being used

    # --- Input Validation ---
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Missing or invalid 'text' field (must be a non-empty string)"}), 400
    if width <= 0 or height <= 0 or max_words <= 0 or scale <= 0:
        return jsonify({"error": "width, height, max_words, and scale must be positive numbers."}), 400
    if language not in ['en', 'zh']:
        return jsonify({"error": "Invalid language specified. Use 'en' or 'zh'."}), 400
    if not isinstance(custom_stopwords, list):
         return jsonify({"error": "custom_stopwords must be a list of strings."}), 400


    # --- Determine and Validate Font Path ---
    # Use font path from request if provided, otherwise use the configured default
    final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
    logging.info(f"Using font path for this request: {final_font_path}")
    # Check if the chosen font path exists *before* processing text
    if not os.path.exists(final_font_path):
        logging.error(f"Font file not found at the selected path: {final_font_path}")
        if font_path_req:
            # If user specified a bad path, inform them
            error_msg = f"Font file specified in request ('{font_path_req}') not found on server."
            return jsonify({"error": error_msg}), 400 # Bad Request
        else:
            # If the default path is bad, it's a server configuration error
            error_msg = f"Default font file ('{DEFAULT_FONT_PATH}') not found or not configured correctly on the server."
            return jsonify({"error": error_msg}), 500 # Internal Server Error

    # --- Process Text to Get Word Frequencies ---
    try:
        logging.info(f"Starting text processing with language mode: {language}")
        word_counts = None
        if language == 'zh':
            if not JIEBA_AVAILABLE:
                logging.error("Cannot process 'zh' language: Jieba library not installed.")
                # Service Unavailable or Not Implemented seems appropriate
                return jsonify({"error": "Server configuration error: Jieba library needed for Chinese processing is missing."}), 503
            # Pass the request-specific custom stopwords
            word_counts = process_text_mixed_chinese_english(raw_text, custom_stopwords)
        elif language == 'en':
             # Pass the request-specific custom stopwords
             word_counts = process_text_nltk(raw_text, custom_stopwords)

        # Check if processing resulted in any words
        if not word_counts:
            logging.warning("No processable words found after text processing and filtering.")
            # Return a specific message indicating no words remained
            return jsonify({"error": "No processable words found in the text after filtering stopwords and short words."}), 400 # Bad Request seems ok

        # Convert Counter object to dictionary for WordCloud library
        frequencies = dict(word_counts)
        actual_word_count = len(frequencies)
        logging.info(f"Text processing complete. Found {actual_word_count} unique words.")

    except RuntimeError as e: # Catch specific errors like Jieba missing if check failed earlier
         logging.error(f"Runtime error during text processing: {e}", exc_info=True)
         return jsonify({"error": f"Text processing runtime error: {e}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error during text processing: {e}", exc_info=True)
        return jsonify({"error": f"Text processing failed: {e}"}), 500

    # --- Create Shape Mask (if requested) ---
    mask_array = None
    if shape != 'rectangle':
        try:
            if shape == 'circle':
                logging.info(f"Creating circular mask with dimensions: {width}x{height}")
                mask_array = create_circle_mask(width, height)
            elif shape == 'ellipse':
                logging.info(f"Creating elliptical mask with dimensions: {width}x{height}")
                mask_array = create_ellipse_mask(width, height)
            else:
                logging.warning(f"Unknown shape '{shape}' requested, defaulting to rectangle (no mask).")
                shape = 'rectangle' # Correct the shape variable if unknown
        except ValueError as e:
            # Catch errors from mask generation (e.g., invalid dimensions)
            logging.error(f"Error creating mask for shape '{shape}': {e}")
            return jsonify({"error": f"Invalid dimensions for mask shape '{shape}': {e}"}), 400
        except Exception as e:
            logging.error(f"Unexpected error creating mask: {e}", exc_info=True)
            return jsonify({"error": "Failed to create shape mask."}), 500

    # --- Generate Word Cloud Image ---
    try:
        logging.info(f"Generating word cloud image (Shape: {shape}, MaxWords: {max_words}, Scale: {scale})...")

        # Instantiate WordCloud object with all configurations
        wordcloud_instance = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            font_path=final_font_path, # Use the validated path
            max_words=max_words,
            mask=mask_array,            # Apply mask if created
            mode=mode,
            scale=scale,
            colormap=colormap_name,
            # Add other potential parameters here if needed in the future
            # e.g., prefer_horizontal, color_func, contour_width, etc.
        )

        # Generate the word positions and rendering from the frequency data
        wordcloud_instance.generate_from_frequencies(frequencies)

        # --- Save Image ---
        # Generate a unique filename using UUID
        filename = f"wc_{uuid.uuid4()}.png" # Always save as PNG for transparency support
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Save the generated word cloud to the file
        wordcloud_instance.to_file(output_path)
        logging.info(f"Word cloud image successfully saved to: {output_path}")

        # --- Prepare and Return Success Response ---
        # Construct the publicly accessible URL for the image
        # Note: This relies on Flask correctly determining the host URL.
        #       Behind a proxy, you might need to configure Flask or the proxy (e.g., X-Forwarded-Proto/Host).
        image_url = f"{request.host_url}images/{filename}"

        return jsonify({
            "imageUrl": image_url,
            "wordCount": actual_word_count, # Number of unique words used
            "shape": shape,
            "dimensions": {"width": width, "height": height}
        }), 200 # OK status

    except FileNotFoundError as e:
         # This specifically catches font file not found *during WordCloud generation* (should be rare if pre-checked)
         logging.error(f"Font file error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Internal Error: Font file '{final_font_path}' became unavailable or is invalid."}), 500
    except ValueError as e:
         # Catches errors from wordcloud lib, e.g., empty frequencies if processing failed silently before
         logging.error(f"Value error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Word cloud generation error: {e}"}), 500 # Use 500 as it's likely internal logic
    except Exception as e:
        # Catch-all for other unexpected errors during generation or saving
        logging.error(f"Unexpected error generating or saving word cloud: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate or save word cloud image: {e}"}), 500

@app.route('/images/<path:filename>')
def get_image(filename):
    """Serves the generated image files from the OUTPUT_DIR."""
    # Security check: Ensure the requested path is within the intended directory
    safe_dir = os.path.abspath(OUTPUT_DIR)
    # Resolve the absolute path for the requested filename within the safe directory
    requested_path = os.path.abspath(os.path.join(safe_dir, filename))

    # Check if the resolved path starts with the safe directory path
    if not requested_path.startswith(safe_dir):
        logging.warning(f"Attempted directory traversal detected for filename: {filename}")
        return "Forbidden", 403 # Prevent access outside OUTPUT_DIR

    # Use Flask's send_from_directory for safer file serving
    try:
        logging.info(f"Serving image file: {filename}")
        # as_attachment=False serves the image inline in the browser if possible
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        logging.warning(f"Requested image file not found: {filename}")
        return "Image not found", 404 # Standard "Not Found" response
    except Exception as e:
        # Catch other potential errors during file serving
        logging.error(f"Error serving image file {filename}: {e}", exc_info=True)
        return "Internal server error", 500

@app.route('/health')
def health_check():
    """Provides a basic health check endpoint for monitoring."""
    # Could add more checks here, e.g., check if font file exists, disk space, etc.
    font_ok = os.path.exists(DEFAULT_FONT_PATH)
    return jsonify({
        "status": "ok",
        "dependencies": {
            "jieba": "available" if JIEBA_AVAILABLE else "missing",
            "nltk": "available" if NLTK_AVAILABLE else "missing/incomplete_data"
        },
        "default_font_found": font_ok
        }), 200

# --- Main Execution Block ---
if __name__ == '__main__':
    # Optional: Initialize Jieba dictionary loading at startup to speed up the first request.
    if JIEBA_AVAILABLE:
        try:
            jieba.initialize()
            logging.info("Jieba initialized successfully.")
        except Exception as e:
            # Log error but don't necessarily stop the server, route handler will check JIEBA_AVAILABLE
            logging.error(f"Failed to initialize Jieba (non-critical at startup): {e}")

    # Perform a critical check for the default font path at startup
    if not os.path.exists(DEFAULT_FONT_PATH):
         logging.error("="*60)
         logging.error(f"CRITICAL STARTUP ERROR: Default font path not found!")
         logging.error(f"Path checked: {DEFAULT_FONT_PATH}")
         logging.error("The service may fail if requests do not provide a valid 'font_path'.")
         logging.error("Please configure DEFAULT_FONT_PATH in the script correctly.")
         logging.error("="*60)
         # Decide if you want to exit if the default font is absolutely mandatory
         # import sys
         # sys.exit("Exiting due to missing default font.")

    # Start the Flask development server.
    # IMPORTANT: For production environments, use a production-grade WSGI server
    # like Gunicorn or uWSGI behind a reverse proxy (like Nginx).
    # Example command for Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 wordcloud_service:app --log-level info
    logging.info("Starting Flask application server...")
    # Set debug=False for production or production-like testing
    # Host '0.0.0.0' makes the server accessible externally (e.g., from Node.js if on different container/machine)
    app.run(host='0.0.0.0', port=5000, debug=False)