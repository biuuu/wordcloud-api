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
# Directory to save generated images
# Uses absolute path relative to this script file
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images')

# --- !!! CRITICAL CONFIGURATION !!! ---
# Default font path. MUST point to a valid font file that supports
# the characters you intend to display (e.g., Chinese AND English).
# Examples:
# Linux: '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
# macOS: '/System/Library/Fonts/PingFang.ttc' or '/Library/Fonts/Arial Unicode MS.ttf'
# Windows: 'C:/Windows/Fonts/msyh.ttc' (Microsoft YaHei) or 'simhei.ttf' (SimHei)
# Use a specific path, avoid relying on font name only unless managed by the system.
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf' # <--- !!! CHANGE THIS !!!
# --- !!! END CRITICAL CONFIGURATION !!! ---

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e:
        logging.error(f"Failed to create output directory {OUTPUT_DIR}: {e}")
        # Depending on severity, you might want to exit here
# ================== END CONFIGURATION ==================


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
        tokens = nltk.word_tokenize(text)
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords:
            stop_words.update(sw.lower() for sw in custom_stopwords)
        words = [
            word for word in tokens
            if word.isalpha() # Keep only actual words (removes punctuation implicitly here)
            and word not in stop_words
            and len(word) > 1 # Remove single-letter words
        ]
    return Counter(words)

def process_text_mixed_chinese_english(text, custom_stopwords=None):
    """
    Processes text containing both Chinese and English using Jieba.
    Keeps English words intact, applies combined stopwords.
    """
    if not JIEBA_AVAILABLE:
        raise RuntimeError("Jieba library is required for Chinese/Mixed processing but not installed.")

    # 1. Cleaning: Lowercase English, remove punctuation and numbers, keep relevant characters.
    text = text.lower()
    # Keep Chinese (\u4e00-\u9fff), English (a-z), and spaces (\s).
    cleaned_text = re.sub(r'[^\u4e00-\u9fffA-Za-z\s]', '', text)
    # Remove potential digits that survived (e.g., if included in \w for other languages)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 2. Define combined stopwords
    # Consider loading a more comprehensive Chinese list from a file
    default_chinese_stopwords = set([
        '的', '了', '和', '是', '就', '都', '而', '及', '與', '或', '个', '也', '这', '那', '之',
        '上', '下', '左', '右', '们', '在', '于', '我', '你', '他', '她', '它', '我们', '你们',
        '他们', '她们', '它们', '吧', '吗', '呢', '啊', '哦', '哈', '嗯', '嗯嗯', '喔', '呗',
        '什么', '没有', '一个', '一些', '这个', '那个', '那个', '这里', '那里', '哪里',
        '这样', '那样', '怎么', '因为', '所以', '但是', '可以', '知道', '觉得', '告诉',
        '现在', '时候', '问题', '一下', '东西', '然后', '还有', '可能', '感觉', '比较',
        '自己', '事情', '需要', '如果', '的话', '就是', '还是', '不过', '而且', '时候',
        '不是', '不要', '不能'
    ])
    combined_stopwords = ENGLISH_STOPWORDS.union(default_chinese_stopwords)
    if custom_stopwords:
        # Ensure custom stopwords are also lowercased if they might be English
        custom_lower = {sw.lower() for sw in custom_stopwords}
        combined_stopwords.update(custom_lower)

    # 3. Tokenization using Jieba
    tokens = jieba.cut(cleaned_text) # Use precise mode (default)

    # 4. Filter tokens
    words = []
    for token in tokens:
        word = token.strip()
        # Filter out: empty strings, stopwords, single characters (Chinese or English), and pure numbers
        if word and word not in combined_stopwords and len(word) > 1 and not word.isdigit():
            words.append(word)

    if not words:
         logging.warning(f"Text after cleaning and tokenization yielded no words. Original text snippet: {text[:100]}...")

    return Counter(words)

# --- End Text Processing Functions ---


# --- Mask Generation Functions ---
def create_circle_mask(width, height):
    """Creates a NumPy array for a circular mask (black circle on white background)."""
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    # Use the smaller dimension for the radius to fit the circle
    radius = min(width, height) // 2
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width] # Create coordinate grid

    # Mask points where distance from center is <= radius
    mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2

    # Create image mask: 255 (white) where no words, 0 (black) where words are allowed
    mask_image = np.full((height, width), 255, dtype=np.uint8)
    mask_image[mask_boolean] = 0 # Set inside circle to black
    return mask_image

def create_ellipse_mask(width, height):
    """Creates a NumPy array for an elliptical mask (black ellipse on white background)."""
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = width // 2, height // 2 # Semi-axes lengths
    y, x = np.ogrid[:height, :width] # Create coordinate grid

    # Mask points inside the ellipse equation
    mask_boolean = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2) <= 1

    # Create image mask: 255 (white) where no words, 0 (black) where words are allowed
    mask_image = np.full((height, width), 255, dtype=np.uint8)
    mask_image[mask_boolean] = 0 # Set inside ellipse to black
    return mask_image
# --- End Mask Generation Functions ---


# --- Flask Routes ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    """
    Main endpoint to generate a word cloud from text.
    Accepts JSON payload with text and various configuration options.
    """
    if not request.is_json:
        logging.warning("Received non-JSON request.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    logging.info(f"Received request data: {data}") # Log incoming data (be careful with sensitive data in production)

    # --- Extract Data and Set Defaults ---
    raw_text = data.get('text')
    language = data.get('language', 'zh').lower() # Default to mixed Chinese/English processing
    custom_stopwords = data.get('custom_stopwords', [])

    # WordCloud visual options
    shape = data.get('shape', 'rectangle').lower()
    width = int(data.get('width', 800))
    height = int(data.get('height', 600))
    # Default background to None, often best for masks with transparency
    background_color = data.get('background_color', None)
    font_path_req = data.get('font_path', None) # Font path from request
    max_words = int(data.get('max_words', 200))
    mode = data.get('mode', 'RGBA') # RGBA recommended for potential transparency
    scale = float(data.get('scale', 1)) # Allow float scale factor

    # --- Input Validation ---
    if not raw_text or not isinstance(raw_text, str):
        return jsonify({"error": "Missing or invalid 'text' field (must be a non-empty string)"}), 400
    if width <= 0 or height <= 0:
        return jsonify({"error": "Width and height must be positive integers"}), 400
    if max_words <= 0:
        return jsonify({"error": "max_words must be a positive integer"}), 400
    if scale <= 0:
        return jsonify({"error": "scale must be a positive number"}), 400
    if language not in ['en', 'zh']:
        return jsonify({"error": "Invalid language specified. Use 'en' or 'zh'."}), 400

    # --- Determine and Validate Font Path ---
    final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
    logging.info(f"Attempting to use font path: {final_font_path}")
    if not os.path.exists(final_font_path):
        logging.error(f"Font file not found at specified or default path: {final_font_path}")
        # If request specified a bad font, don't fallback automatically, inform user
        if font_path_req:
             error_msg = f"Font file specified in request ('{font_path_req}') not found on server."
             return jsonify({"error": error_msg}), 400
        else:
             # If default font is bad, it's a server configuration issue
             error_msg = f"Default font file ('{DEFAULT_FONT_PATH}') not found or not configured correctly on the server."
             return jsonify({"error": error_msg}), 500 # Internal Server Error seems appropriate

    # --- Process Text ---
    try:
        logging.info(f"Processing text with language mode: {language}")
        word_counts = None
        if language == 'zh':
            if not JIEBA_AVAILABLE:
                logging.error("Cannot process 'zh' language: Jieba library not installed.")
                return jsonify({"error": "Server configuration error: Jieba library needed for Chinese processing is missing."}), 501
            word_counts = process_text_mixed_chinese_english(raw_text, custom_stopwords)
        elif language == 'en':
             # NLTK availability check is handled inside the function
             word_counts = process_text_nltk(raw_text, custom_stopwords)

        if not word_counts:
            logging.warning("No processable words found after filtering.")
            return jsonify({"error": "No processable words found in the text after filtering stopwords and short words."}), 400

        frequencies = dict(word_counts)
        actual_word_count = len(frequencies)
        logging.info(f"Found {actual_word_count} unique words after processing.")

    except Exception as e:
        logging.error(f"Error during text processing: {e}", exc_info=True)
        return jsonify({"error": f"Text processing failed: {e}"}), 500

    # --- Create Shape Mask (if requested) ---
    mask_array = None
    try:
        if shape == 'circle':
            logging.info(f"Creating circular mask with dimensions: {width}x{height}")
            mask_array = create_circle_mask(width, height)
        elif shape == 'ellipse':
            logging.info(f"Creating elliptical mask with dimensions: {width}x{height}")
            mask_array = create_ellipse_mask(width, height)
        elif shape != 'rectangle':
             logging.warning(f"Unknown shape '{shape}', defaulting to rectangle (no mask).")
        # If shape is 'rectangle', mask_array remains None
    except ValueError as e:
         logging.error(f"Error creating mask: {e}")
         return jsonify({"error": f"Invalid dimensions for mask: {e}"}), 400


    # --- Generate Word Cloud Image ---
    try:
        logging.info("Generating word cloud image...")
        wordcloud_instance = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            font_path=final_font_path, # Use the validated path
            max_words=max_words,
            mask=mask_array,            # Apply mask if created
            mode=mode,
            scale=scale,
            # Other potential options you might want to expose via API:
            # prefer_horizontal=data.get('prefer_horizontal', 0.9),
            # color_func=..., # More complex color logic
            # contour_width=float(data.get('contour_width', 0)),
            # contour_color=data.get('contour_color', 'black'),
            # collocations=bool(data.get('collocations', True)), # Find word pairs
            # collocation_threshold=int(data.get('collocation_threshold', 30)),
            # min_font_size=int(data.get('min_font_size', 4)),
            # max_font_size=data.get('max_font_size', None), # Let library decide max based on size/mask
        )

        # Generate the word cloud from calculated frequencies
        wordcloud_instance.generate_from_frequencies(frequencies)

        # --- Save Image ---
        filename = f"wc_{uuid.uuid4()}.png" # Use PNG for transparency support
        output_path = os.path.join(OUTPUT_DIR, filename)
        wordcloud_instance.to_file(output_path)
        logging.info(f"Word cloud image saved to: {output_path}")

        # --- Prepare Response ---
        # Construct the URL assuming the service base URL can be derived from the request
        # Note: Behind a proxy, request.host_url might need adjustment (e.g., using X-Forwarded-Host)
        image_url = f"{request.host_url}images/{filename}"

        return jsonify({
            "imageUrl": image_url,
            "wordCount": actual_word_count, # Number of unique words placed
            "shape": shape,
            "dimensions": {"width": width, "height": height}
        })

    except FileNotFoundError as e:
         # This specifically catches font file not found *during WordCloud generation*
         logging.error(f"Font file error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Internal Error: Font file '{final_font_path}' became unavailable or invalid."}), 500
    except ValueError as e:
         # Catches errors like empty frequencies if processing failed silently before
         logging.error(f"Value error during WordCloud generation: {e}", exc_info=True)
         return jsonify({"error": f"Generation error: {e}"}), 500
    except Exception as e:
        # Catch-all for other unexpected errors during generation or saving
        logging.error(f"Unexpected error generating or saving word cloud: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate word cloud image: {e}"}), 500


@app.route('/images/<path:filename>')
def get_image(filename):
    """Serves the generated image file."""
    # Basic security check to prevent directory traversal
    safe_dir = os.path.abspath(OUTPUT_DIR)
    safe_path = os.path.abspath(os.path.join(safe_dir, filename))

    if not safe_path.startswith(safe_dir):
        logging.warning(f"Attempted directory traversal: {filename}")
        return "Forbidden", 403

    # Use send_from_directory for safer file serving
    try:
        logging.info(f"Serving image: {filename}")
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=False) # Serve inline
    except FileNotFoundError:
        logging.warning(f"Image not found: {filename}")
        return "Image not found", 404
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}", exc_info=True)
        return "Internal server error", 500

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    # Could add checks here (e.g., disk space, font file exists)
    return jsonify({"status": "ok"}), 200

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize Jieba dictionary loading (optional but can speed up first request)
    if JIEBA_AVAILABLE:
        try:
            jieba.initialize()
            logging.info("Jieba initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Jieba: {e}")

    # Check if critical default font path exists at startup
    if not os.path.exists(DEFAULT_FONT_PATH):
         logging.error("="*60)
         logging.error(f"CRITICAL ERROR: Default font path not found: {DEFAULT_FONT_PATH}")
         logging.error("The service WILL NOT WORK correctly without a valid font.")
         logging.error("Please configure DEFAULT_FONT_PATH in the script.")
         logging.error("="*60)
         # Consider exiting if the default font is essential and missing:
         # import sys
         # sys.exit(1)

    # Run Flask's built-in development server
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 wordcloud_service:app
    logging.info("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production-like behavior