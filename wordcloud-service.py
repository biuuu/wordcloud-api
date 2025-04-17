#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import logging
import re
from collections import Counter
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
    logging.error("Jieba library or analyse submodule not found. Service cannot process text.", exc_info=True)
    # Consider exiting if Jieba is absolutely required
    # import sys
    # sys.exit("Exiting: Jieba library is mandatory and was not found.")

# --- End Text Processing Imports ---


# ================== CONFIGURATION ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_images')
# --- !!! CRITICAL CONFIGURATION !!! ---
# Default font path. MUST point to a valid font file supporting Chinese.
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf'
# --- !!! END CRITICAL CONFIGURATION !!! ---
STOPWORDS_FILENAME = 'baidu_stopwords.txt' # Or your preferred filename
STOPWORDS_FILE_PATH = os.path.join(SCRIPT_DIR, STOPWORDS_FILENAME)

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    try: os.makedirs(OUTPUT_DIR); logging.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e: logging.error(f"Failed to create output directory {OUTPUT_DIR}: {e}", exc_info=True)
# ================== END CONFIGURATION ==================

# --- Load Chinese Stopwords At Startup ---
LOADED_CHINESE_STOPWORDS = set()
DEFAULT_HARDCODED_CHINESE_STOPWORDS = set(['的', '了', '和', '是']) # Minimal fallback
try:
    logging.info(f"Attempting to load Chinese stopwords from: {STOPWORDS_FILE_PATH}")
    with open(STOPWORDS_FILE_PATH, 'r', encoding='utf-8') as f:
        LOADED_CHINESE_STOPWORDS = {line.strip() for line in f if line.strip()}
    if LOADED_CHINESE_STOPWORDS: logging.info(f"Successfully loaded {len(LOADED_CHINESE_STOPWORDS)} Chinese stopwords from {STOPWORDS_FILENAME}.")
    else: logging.warning(f"{STOPWORDS_FILENAME} is empty, using fallback."); LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except FileNotFoundError:
    logging.warning(f"Stopword file '{STOPWORDS_FILENAME}' not found at {STOPWORDS_FILE_PATH}. Falling back to default list.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except Exception as e:
    logging.error(f"An error occurred while loading stopwords from {STOPWORDS_FILENAME}: {e}", exc_info=True)
    logging.warning("Falling back to default hardcoded Chinese stopwords due to loading error.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
finally:
     logging.info(f"Using {len(LOADED_CHINESE_STOPWORDS)} Chinese stopwords for processing.")
# --- End Stopword Loading ---


# Initialize Flask application
app = Flask(__name__)


# --- Text Processing Function (Frequency Count) ---
def process_text_chinese_frequency(text, combined_stopwords):
    """
    Processes Chinese text using Jieba frequency count method.
    Assumes user dictionary words have already been added via jieba.add_word().
    Uses the provided combined_stopwords set for filtering.
    """
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required but not available.")

    # 1. Cleaning: Keep only Chinese characters and spaces.
    #    You might want to adjust this if you need to keep alphanumeric for mixed content.
    #    This version focuses purely on Chinese as requested.
    cleaned_text = re.sub(r'[^\u4e00-\u9fff\s]', '', text) # Keep only Chinese chars and spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Normalize spaces

    # 2. Tokenization using Jieba (respects added user words)
    tokens = jieba.cut(cleaned_text)

    # 3. Filter tokens based on stopwords and length
    words = []
    for token in tokens:
        word = token.strip()
        # Filter out: empty strings, stopwords, single characters
        if word and word not in combined_stopwords and len(word) > 1:
            words.append(word)

    if not words: logging.warning(f"Frequency method yielded no words after processing and filtering.")
    # Return a frequency count of the valid words
    return Counter(words)
# --- End Text Processing Function ---


# --- Mask Generation Functions ---
def create_circle_mask(width, height):
    if width <= 0 or height <= 0: raise ValueError("Mask dimensions must be positive.")
    radius = min(width, height) // 2; center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]; mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image

def create_ellipse_mask(width, height):
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
    Generates Chinese word cloud. Accepts 'extraction_method' ('frequency' or 'tfidf'),
    'user_dict_words', 'custom_stopwords', and visual parameters.
    """
    if not JIEBA_AVAILABLE: # Check if Jieba loaded correctly at startup
        logging.error("Jieba not available, cannot process request.")
        return jsonify({"error": "Server configuration error: Jieba library is missing or failed to load."}), 503

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    logging.info(f"Received request data (keys): {list(data.keys())}")

    # --- Extract Data and Set Defaults ---
    raw_text = data.get('text')
    custom_stopwords = data.get('custom_stopwords', [])
    user_dict_words = data.get('user_dict_words', [])
    extraction_method = data.get('extraction_method', 'frequency').lower()

    shape = data.get('shape', 'rectangle').lower()
    try:
        width = int(data.get('width', 800)); height = int(data.get('height', 600))
        max_words = int(data.get('max_words', 200)); scale = float(data.get('scale', 1))
        top_k = int(data.get('top_k', max_words)) # For TF-IDF
    except (ValueError, TypeError): return jsonify({"error": "width, height, max_words, top_k must be integers; scale must be a number."}), 400
    background_color = data.get('background_color', None)
    font_path_req = data.get('font_path', None)
    mode = data.get('mode', 'RGBA'); colormap_name = data.get('colormap', 'tab20')

    # --- Input Validation ---
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip(): return jsonify({"error": "Missing or invalid 'text'"}), 400
    if width <= 0 or height <= 0 or max_words <= 0 or scale <= 0 or top_k <= 0: return jsonify({"error": "width, height, max_words, scale, top_k must be positive."}), 400
    if extraction_method not in ['frequency', 'tfidf']: return jsonify({"error": "Invalid extraction_method. Use 'frequency' or 'tfidf'."}), 400
    if not isinstance(custom_stopwords, list): return jsonify({"error": "custom_stopwords must be a list."}), 400
    if not isinstance(user_dict_words, list): return jsonify({"error": "user_dict_words must be a list."}), 400

    # --- Font Path Validation ---
    final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
    if not os.path.exists(final_font_path):
        logging.error(f"Font file not found: {final_font_path}")
        if font_path_req: return jsonify({"error": f"Requested font '{font_path_req}' not found."}), 400
        else: return jsonify({"error": f"Default font '{DEFAULT_FONT_PATH}' not found."}), 500

    frequencies = {}
    actual_word_count = 0

    try:
        # --- Apply User Dictionary (BEFORE processing) ---
        # This affects both jieba.cut and jieba.analyse
        if user_dict_words:
            added_count = 0
            for word in user_dict_words:
                if isinstance(word, str) and word.strip():
                    jieba.add_word(word.strip()); added_count += 1
            if added_count > 0: logging.info(f"Temporarily added {added_count} custom words to Jieba dictionary.")

        # --- Prepare Combined Stopwords (used for filtering in both methods) ---
        combined_stopwords = LOADED_CHINESE_STOPWORDS.copy() # Start with base file content
        if custom_stopwords:
            combined_stopwords.update(sw for sw in custom_stopwords if isinstance(sw, str)) # Add request-specific
        logging.debug(f"Total combined stopwords count for filtering: {len(combined_stopwords)}")

        # --- Select Extraction Method ---
        logging.info(f"Processing text using method: '{extraction_method}'")

        if extraction_method == 'tfidf':
            # TF-IDF Method: Extract keywords, then filter stopwords manually
            logging.debug(f"Calling extract_tags with topK={top_k}")
            # Extract raw keywords without internal stopword filtering via parameter
            raw_keywords = jieba.analyse.extract_tags(
                raw_text, topK=top_k, withWeight=True, allowPOS=()
            )
            logging.debug(f"Raw keywords from TF-IDF: {raw_keywords[:10]}...") # Log first few

            if not raw_keywords:
                logging.warning("TF-IDF extraction returned no raw keywords.")
                return jsonify({"error": "No keywords found using TF-IDF before stopword filtering."}), 400

            # Filter results manually using the combined stopword list
            tfidf_scale_factor = 1000 # Adjust as needed
            frequencies = {}
            for word, weight in raw_keywords:
                word_cleaned = word.strip()
                # Apply filters: not in combined_stopwords, length > 1
                if word_cleaned and word_cleaned not in combined_stopwords and len(word_cleaned) > 1:
                    frequencies[word_cleaned] = max(1, int(weight * tfidf_scale_factor))

            if not frequencies:
                 logging.warning("TF-IDF extraction yielded no keywords after filtering stopwords.")
                 return jsonify({"error": "No valid keywords remained after filtering stopwords from TF-IDF results."}), 400

            actual_word_count = len(frequencies)
            logging.info(f"TF-IDF extraction complete. Found {actual_word_count} keywords after filtering.")

        else: # Default to 'frequency' method
            # Frequency Method: Use jieba.cut and filter using combined stopwords
            word_counts = process_text_chinese_frequency(raw_text, combined_stopwords)

            if not word_counts: return jsonify({"error": "No processable words found after frequency counting."}), 400
            frequencies = dict(word_counts)
            actual_word_count = len(frequencies)
            logging.info(f"Frequency counting complete. Found {actual_word_count} unique words.")

    except RuntimeError as e:
         logging.error(f"Runtime error during text processing/extraction: {e}", exc_info=True)
         return jsonify({"error": f"Text processing runtime error: {e}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error during text processing/extraction: {e}", exc_info=True)
        return jsonify({"error": f"Text processing/extraction failed: {e}"}), 500

    # Final check if frequencies is empty before generating cloud
    if not frequencies:
         logging.error(f"Frequencies dictionary is empty after method '{extraction_method}'. Cannot generate cloud.")
         return jsonify({"error": f"Failed to extract any valid words using method '{extraction_method}'."}), 500

    # --- Create Shape Mask (if requested) ---
    mask_array = None
    if shape != 'rectangle':
        # (Mask creation logic remains the same)
        try:
            if shape == 'circle': mask_array = create_circle_mask(width, height)
            elif shape == 'ellipse': mask_array = create_ellipse_mask(width, height)
            else: logging.warning(f"Unknown shape '{shape}', using rectangle."); shape = 'rectangle'
        except ValueError as e: return jsonify({"error": f"Invalid dimensions for mask: {e}"}), 400
        except Exception as e: logging.error(f"Mask creation error: {e}", exc_info=True); return jsonify({"error": "Failed to create shape mask."}), 500

    # --- Generate Word Cloud Image ---
    try:
        logging.info(f"Generating word cloud image (Method: {extraction_method}, Shape: {shape}, MaxWords: {max_words})...")
        wordcloud_instance = WordCloud(
            width=width, height=height, background_color=background_color,
            font_path=final_font_path, max_words=max_words, mask=mask_array,
            mode=mode, scale=scale, colormap=colormap_name,
        )
        wordcloud_instance.generate_from_frequencies(frequencies)

        # --- Save Image ---
        filename = f"wc_{uuid.uuid4()}.png"; output_path = os.path.join(OUTPUT_DIR, filename)
        wordcloud_instance.to_file(output_path)
        logging.info(f"Word cloud image saved: {output_path}")

        # --- Prepare Response ---
        image_url = f"{request.host_url}images/{filename}"
        return jsonify({
            "imageUrl": image_url, "wordCount": min(len(wordcloud_instance.words_), max_words), # Use actual words placed by lib
            "shape": shape, "dimensions": {"width": width, "height": height},
            "colormap_used": colormap_name, "extraction_method_used": extraction_method
        }), 200

    # --- Error Handling ---
    except FileNotFoundError as e: logging.error(f"Font error during WC generation: {e}", exc_info=True); return jsonify({"error": f"Internal Error: Font unavailable."}), 500
    except ValueError as e: logging.error(f"Value error during WC generation: {e}", exc_info=True); return jsonify({"error": f"Generation error: {e}"}), 500
    except Exception as e: logging.error(f"Unexpected error generating/saving WC: {e}", exc_info=True); return jsonify({"error": f"Failed to generate/save image: {e}"}), 500


@app.route('/images/<path:filename>')
def get_image(filename):
    # (Serving logic remains the same)
    safe_dir = os.path.abspath(OUTPUT_DIR); requested_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not requested_path.startswith(safe_dir): return "Forbidden", 403
    try: return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)
    except FileNotFoundError: return "Image not found", 404
    except Exception as e: logging.error(f"Error serving file {filename}: {e}"); return "Internal server error", 500

@app.route('/health')
def health_check():
    # Simplified health check (removed NLTK)
    font_ok = os.path.exists(DEFAULT_FONT_PATH)
    return jsonify({
        "status": "ok",
        "dependencies": { "jieba": "available" if JIEBA_AVAILABLE else "missing" },
        "default_font_found": font_ok
        }), 200

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize Jieba only if available
    if JIEBA_AVAILABLE:
        try:
            jieba.initialize()
            logging.info("Jieba initialized.")
            # Set the base stop words file for TF-IDF *at startup* if you want
            # Note: This will be the *only* stopword list TF-IDF uses internally
            if STOPWORDS_FILE_PATH and os.path.exists(STOPWORDS_FILE_PATH):
                jieba.analyse.set_stop_words(STOPWORDS_FILE_PATH)
                logging.info(f"Set TF-IDF base stopwords file: {STOPWORDS_FILE_PATH}")
            else:
                logging.warning("Base stopword file for TF-IDF not found or not specified, TF-IDF might use default or none.")
            # **Decision**: We are filtering *after* extract_tags, so we don't call set_stop_words here.
        except Exception as e:
            logging.error(f"Failed to initialize Jieba: {e}")

    # Check default font path critically at startup
    if not os.path.exists(DEFAULT_FONT_PATH):
         logging.error("="*60 + f"\nCRITICAL STARTUP ERROR: Default font path not found: {DEFAULT_FONT_PATH}\n" + "="*60)

    logging.info("Starting Flask application server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Remember debug=False for production