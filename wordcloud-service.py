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
    import jieba.analyse # <--- Import analyse submodule
    JIEBA_AVAILABLE = True
    logging.info("Jieba library and analyse submodule found.")
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("Jieba library or analyse submodule not found. TF-IDF and Chinese processing will fail.")

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    logging.info("NLTK library and data found.")
except (ImportError, LookupError) as e:
    NLTK_AVAILABLE = False
    ENGLISH_STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    logging.warning(f"NLTK not available ({e}). Using basic English stopwords.")
# --- End Text Processing Imports ---


# ================== CONFIGURATION ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_images')
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf'
STOPWORDS_FILENAME = 'baidu_stopwords.txt'
STOPWORDS_FILE_PATH = os.path.join(SCRIPT_DIR, STOPWORDS_FILENAME)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if not os.path.exists(OUTPUT_DIR):
    try: os.makedirs(OUTPUT_DIR); logging.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e: logging.error(f"Failed to create output directory {OUTPUT_DIR}: {e}", exc_info=True)
# ================== END CONFIGURATION ==================

# --- Load Stopwords At Startup ---
LOADED_CHINESE_STOPWORDS = set()
DEFAULT_HARDCODED_CHINESE_STOPWORDS = set(['的', '了', '和', '是']) # Minimal fallback
try:
    with open(STOPWORDS_FILE_PATH, 'r', encoding='utf-8') as f:
        LOADED_CHINESE_STOPWORDS = {line.strip() for line in f if line.strip()}
    if LOADED_CHINESE_STOPWORDS: logging.info(f"Loaded {len(LOADED_CHINESE_STOPWORDS)} stopwords from {STOPWORDS_FILENAME}.")
    else: logging.warning(f"{STOPWORDS_FILENAME} is empty, using fallback."); LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except FileNotFoundError:
    logging.warning(f"{STOPWORDS_FILENAME} not found, using fallback."); LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except Exception as e:
    logging.error(f"Error loading stopwords: {e}", exc_info=True); LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
# --- End Stopword Loading ---

app = Flask(__name__)

# --- Text Processing Functions ---
# NOTE: User dictionary words are now added in the main route *before* calling these

def process_text_nltk(text, custom_stopwords=None):
    """Process primarily English text using NLTK (if available) or basic split."""
    # (Implementation remains the same as before)
    if not NLTK_AVAILABLE:
        logging.warning("NLTK unavailable, falling back to basic split for tokenization.")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text); text = re.sub(r'\d+', '', text)
        tokens = text.split()
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords: stop_words.update(sw.lower() for sw in custom_stopwords)
        words = [word for word in tokens if word and word not in stop_words and len(word) > 1]
    else:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords: stop_words.update(sw.lower() for sw in custom_stopwords)
        words = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]
    return Counter(words)

def process_text_mixed_chinese_english(text, combined_stopwords):
    """
    Processes text containing both Chinese and English using Jieba (Frequency method).
    Assumes user dictionary words have already been added via jieba.add_word().
    Uses the provided combined_stopwords set.
    """
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required but not installed.")

    # 1. Cleaning (same as before)
    text = text.lower()
    cleaned_text = re.sub(r'[^\u4e00-\u9fffA-Za-z\s]', '', text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 2. Tokenization using Jieba (respects added user words)
    tokens = jieba.cut(cleaned_text)

    # 3. Filter tokens
    words = []
    for token in tokens:
        word = token.strip()
        # Filter using the pre-combined stopword list
        if word and word not in combined_stopwords and len(word) > 1 and not word.isdigit():
            words.append(word)

    if not words: logging.warning(f"Frequency method yielded no words...")
    return Counter(words)

# --- End Text Processing Functions ---


# --- Mask Generation Functions ---
# (create_circle_mask and create_ellipse_mask remain the same)
def create_circle_mask(width, height):
    if width <= 0 or height <= 0: raise ValueError("Mask dimensions must be positive.")
    radius = min(width, height) // 2; center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image

def create_ellipse_mask(width, height):
    if width <= 0 or height <= 0: raise ValueError("Mask dimensions must be positive.")
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = max(1, width // 2), max(1, height // 2)
    y, x = np.ogrid[:height, :width]
    mask_boolean = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2) <= 1
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image
# --- End Mask Generation Functions ---


# --- Flask Routes ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    """
    Generates word cloud. Accepts 'extraction_method' ('frequency' or 'tfidf'),
    'user_dict_words', and other visual parameters.
    """
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    logging.info(f"Received request data (keys): {list(data.keys())}")

    # --- Extract Data and Set Defaults ---
    raw_text = data.get('text')
    language = data.get('language', 'zh').lower()
    custom_stopwords = data.get('custom_stopwords', [])
    user_dict_words = data.get('user_dict_words', [])
    extraction_method = data.get('extraction_method', 'frequency').lower() # New parameter

    # Visual options
    shape = data.get('shape', 'rectangle').lower()
    try:
        width = int(data.get('width', 800)); height = int(data.get('height', 600))
        max_words = int(data.get('max_words', 200)) # Used as general limit and default top_k
        scale = float(data.get('scale', 1))
        # top_k specifically for TF-IDF, defaults to max_words if not provided
        top_k = int(data.get('top_k', max_words))
    except (ValueError, TypeError): return jsonify({"error": "width, height, max_words, top_k must be integers; scale must be a number."}), 400
    background_color = data.get('background_color', None)
    font_path_req = data.get('font_path', None)
    mode = data.get('mode', 'RGBA')
    colormap_name = data.get('colormap', 'tab20')

    # --- Input Validation ---
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip(): return jsonify({"error": "Missing or invalid 'text'"}), 400
    if width <= 0 or height <= 0 or max_words <= 0 or scale <= 0 or top_k <= 0: return jsonify({"error": "width, height, max_words, scale, top_k must be positive."}), 400
    if language not in ['en', 'zh']: return jsonify({"error": "Invalid language. Use 'en' or 'zh'."}), 400
    if extraction_method not in ['frequency', 'tfidf']: return jsonify({"error": "Invalid extraction_method. Use 'frequency' or 'tfidf'."}), 400
    if not isinstance(custom_stopwords, list): return jsonify({"error": "custom_stopwords must be a list."}), 400
    if not isinstance(user_dict_words, list): return jsonify({"error": "user_dict_words must be a list."}), 400
    # TF-IDF extraction currently implemented only for Chinese
    if extraction_method == 'tfidf' and language != 'zh':
        return jsonify({"error": "TF-IDF extraction method is currently only supported for language='zh'."}), 400


    # --- Font Path Validation ---
    final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
    if not os.path.exists(final_font_path):
        # ... (Font validation logic remains the same) ...
        logging.error(f"Font file not found: {final_font_path}")
        if font_path_req: return jsonify({"error": f"Requested font '{font_path_req}' not found."}), 400
        else: return jsonify({"error": f"Default font '{DEFAULT_FONT_PATH}' not found."}), 500

    frequencies = {}
    actual_word_count = 0

    try:
        # --- Apply User Dictionary (if needed) ---
        # Add user words *before* any jieba processing (cut or analyse)
        if language == 'zh' and JIEBA_AVAILABLE and user_dict_words:
            added_count = 0
            for word in user_dict_words:
                if isinstance(word, str) and word.strip():
                    jieba.add_word(word.strip())
                    added_count += 1
            if added_count > 0:
                logging.info(f"Temporarily added {added_count} custom words to Jieba dictionary for this request.")

        # --- Prepare Combined Stopwords ---
        # Applicable for both frequency and TF-IDF methods
        combined_stopwords = ENGLISH_STOPWORDS.copy() # Start with English
        if language == 'zh':
            combined_stopwords.update(LOADED_CHINESE_STOPWORDS) # Add loaded Chinese
        if custom_stopwords:
            combined_stopwords.update(sw.lower() for sw in custom_stopwords) # Add request-specific
        logging.debug(f"Total combined stopwords count: {len(combined_stopwords)}")


        # --- Select Extraction Method ---
        logging.info(f"Processing text using method: '{extraction_method}' (lang: {language})")

        if extraction_method == 'tfidf':
            if not JIEBA_AVAILABLE:
                 return jsonify({"error": "Server config error: Jieba needed for TF-IDF."}), 503

            # Use jieba.analyse.extract_tags
            # withWeight=True returns (word, weight) tuples
            # Pass combined stopwords list to the analyser
            keywords = jieba.analyse.extract_tags(
                raw_text,
                topK=top_k,
                withWeight=True,
                allowPOS=(), # No POS filtering for now
                stopwords=combined_stopwords # Pass the combined set
            )

            if not keywords:
                logging.warning("TF-IDF extraction returned no keywords.")
                return jsonify({"error": "No keywords found using TF-IDF after filtering."}), 400

            # Convert list of (word, weight) tuples to dict {word: scaled_weight}
            # Scale weights because TF-IDF scores are often small floats
            # Scaling makes them more visually distinct in the word cloud
            # Adjust the scaling factor as needed based on visual results
            tfidf_scale_factor = 1000
            frequencies = {word: max(1, int(weight * tfidf_scale_factor)) for word, weight in keywords if len(word) > 1} # Ensure weight is at least 1 after scaling and filter short words again just in case
            actual_word_count = len(frequencies)
            logging.info(f"TF-IDF extraction complete. Found {actual_word_count} keywords.")

        else: # Default to 'frequency' method
            word_counts = None
            if language == 'zh':
                if not JIEBA_AVAILABLE:
                     return jsonify({"error": "Server config error: Jieba missing for frequency count."}), 503
                # Pass only the combined stopwords, user dict applied earlier
                word_counts = process_text_mixed_chinese_english(raw_text, combined_stopwords)
            elif language == 'en':
                 word_counts = process_text_nltk(raw_text, custom_stopwords) # NLTK handles its own stopwords

            if not word_counts:
                 return jsonify({"error": "No processable words found after frequency counting and filtering."}), 400

            # Convert Counter to dict for WordCloud input
            # No need to limit by max_words here, WordCloud's max_words parameter handles the final limit.
            frequencies = dict(word_counts)
            actual_word_count = len(frequencies)
            logging.info(f"Frequency counting complete. Found {actual_word_count} unique words.")

    except RuntimeError as e:
         logging.error(f"Runtime error during text processing/extraction: {e}", exc_info=True)
         return jsonify({"error": f"Text processing runtime error: {e}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error during text processing/extraction: {e}", exc_info=True)
        return jsonify({"error": f"Text processing/extraction failed: {e}"}), 500

    # Double-check if frequencies dictionary is empty after processing
    if not frequencies:
         logging.error(f"Frequencies dictionary is empty after method '{extraction_method}'. Cannot generate cloud.")
         return jsonify({"error": f"Failed to extract any valid words using method '{extraction_method}'."}), 500


    # --- Create Shape Mask (if requested) ---
    mask_array = None
    if shape != 'rectangle':
        # ... (Mask creation logic remains the same) ...
        try:
            if shape == 'circle': mask_array = create_circle_mask(width, height)
            elif shape == 'ellipse': mask_array = create_ellipse_mask(width, height)
            else: logging.warning(f"Unknown shape '{shape}', using rectangle."); shape = 'rectangle'
        except ValueError as e: return jsonify({"error": f"Invalid dimensions for mask: {e}"}), 400
        except Exception as e: logging.error(f"Mask creation error: {e}", exc_info=True); return jsonify({"error": "Failed to create shape mask."}), 500


    # --- Generate Word Cloud Image ---
    try:
        logging.info(f"Generating word cloud image (Method: {extraction_method}, Shape: {shape}, Colormap: {colormap_name}, MaxWords: {max_words})...")

        wordcloud_instance = WordCloud(
            width=width, height=height, background_color=background_color,
            font_path=final_font_path,
            max_words=max_words, # Let WordCloud handle the final word limit
            mask=mask_array,
            mode=mode, scale=scale, colormap=colormap_name,
        )
        # Generate using the prepared frequencies dictionary
        wordcloud_instance.generate_from_frequencies(frequencies)

        # --- Save Image ---
        filename = f"wc_{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        wordcloud_instance.to_file(output_path)
        logging.info(f"Word cloud image saved: {output_path}")

        # --- Prepare Response ---
        image_url = f"{request.host_url}images/{filename}"
        return jsonify({
            "imageUrl": image_url,
            "wordCount": min(actual_word_count, max_words), # Actual words placed might be <= max_words
            "shape": shape,
            "dimensions": {"width": width, "height": height},
            "colormap_used": colormap_name,
            "extraction_method_used": extraction_method
        }), 200

    # --- Error Handling ---
    except FileNotFoundError as e: logging.error(f"Font error during WC generation: {e}", exc_info=True); return jsonify({"error": f"Internal Error: Font unavailable."}), 500
    except ValueError as e: logging.error(f"Value error during WC generation (e.g., empty frequencies?): {e}", exc_info=True); return jsonify({"error": f"Generation error: {e}"}), 500
    except Exception as e: logging.error(f"Unexpected error generating/saving WC: {e}", exc_info=True); return jsonify({"error": f"Failed to generate/save image: {e}"}), 500


@app.route('/images/<path:filename>')
def get_image(filename):
    # ... (Serving logic remains the same) ...
    safe_dir = os.path.abspath(OUTPUT_DIR); requested_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not requested_path.startswith(safe_dir): return "Forbidden", 403
    try: return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)
    except FileNotFoundError: return "Image not found", 404
    except Exception as e: logging.error(f"Error serving file {filename}: {e}"); return "Internal server error", 500

@app.route('/health')
def health_check():
    # ... (Health check remains the same) ...
    font_ok = os.path.exists(DEFAULT_FONT_PATH)
    return jsonify({ "status": "ok", "dependencies": {"jieba": "available" if JIEBA_AVAILABLE else "missing", "nltk": "available" if NLTK_AVAILABLE else "missing/incomplete_data"}, "default_font_found": font_ok }), 200

# --- Main Execution ---
if __name__ == '__main__':
    # (Jieba init and font check remain the same)
    if JIEBA_AVAILABLE:
        try: jieba.initialize(); logging.info("Jieba initialized.")
        except Exception as e: logging.error(f"Failed to initialize Jieba: {e}")
    if not os.path.exists(DEFAULT_FONT_PATH):
         logging.error("="*60 + f"\nCRITICAL STARTUP ERROR: Default font path not found: {DEFAULT_FONT_PATH}\n" + "="*60)

    logging.info("Starting Flask application server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Remember debug=False for production