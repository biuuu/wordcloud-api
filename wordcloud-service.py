import os
import uuid
import logging
import re
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from wordcloud import WordCloud

# --- Text Processing Imports ---
try:
    import jieba
    JIEBA_AVAILABLE = True
    logging.info("Jieba library found, Chinese tokenization enabled.")
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("Jieba library not found. Chinese tokenization will NOT work unless specified otherwise.")

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find('tokenizers/punkt') # Needed for word_tokenize
    nltk.data.find('corpora/stopwords')
    NLTK_AVAILABLE = True
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    logging.info("NLTK library and data found, English tokenization/stopwords enabled.")
except (ImportError, LookupError) as e:
    NLTK_AVAILABLE = False
    # Define a basic English stopword list if NLTK is unavailable
    ENGLISH_STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    logging.warning(f"NLTK library/data not available or configured correctly ({e}). Using basic English stopword list.")

# --- End Text Processing Imports ---

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_images')
DEFAULT_FONT_PATH = '/path/to/your/font.ttf' # <--- !!! MUST CONFIGURE: Needs to support BOTH Chinese and English Glyphs !!!
# --- Configuration End ---

logging.basicConfig(level=logging.INFO)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

app = Flask(__name__)

# --- Text Processing Functions ---

def process_text_nltk(text, custom_stopwords=None):
    """Process primarily English text using NLTK (if available) or basic split."""
    # (Keep this function as before for the language='en' case)
    if not NLTK_AVAILABLE:
        logging.warning("NLTK unavailable, falling back to basic split for tokenization.")
        # Basic cleaning: lowercase, remove non-alphanumeric, split
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\d+', '', text)    # Remove numbers
        tokens = text.split()
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        words = [word for word in tokens if word and word not in stop_words and len(word) > 1]
    else:
        # NLTK processing
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        stop_words = ENGLISH_STOPWORDS.copy()
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        words = [
            word for word in tokens
            if word.isalpha() # Keep only actual words
            and word not in stop_words
            and len(word) > 1 # Remove single-letter words (optional)
        ]
    return Counter(words)

def process_text_mixed_chinese_english(text, custom_stopwords=None):
    """
    Processes text containing both Chinese and English.
    Uses Jieba for Chinese segmentation, keeps English words intact.
    Removes punctuation and numbers, applies combined stopwords.
    """
    if not JIEBA_AVAILABLE:
        raise RuntimeError("Jieba library is required for Chinese/Mixed processing but not installed.")

    # 1. Cleaning: Lowercase English, remove punctuation and numbers. Keep Chinese chars, English chars, and spaces.
    text = text.lower() # Lowercase for consistent English word handling
    # Keep Chinese (\u4e00-\u9fff), English (a-z), and spaces (\s). Remove others.
    cleaned_text = re.sub(r'[^\u4e00-\u9fffA-Za-z\s]', '', text)
    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 2. Define combined stopwords
    # Use a basic default Chinese stopword list (consider using a file for a better list)
    default_chinese_stopwords = set([
        '的', '了', '和', '是', '就', '都', '而', '及', '與', '或', '个', '也', '这', '那', '之',
        '上', '下', '左', '右', '们', '在', '于', '我', '你', '他', '她', '它', '我们', '你们',
        '他们', '她们', '它们', '吧', '吗', '呢', '啊', '哦', '哈', '嗯', '嗯嗯', # Common chat particles
        '什么', '没有', '一个', '一些', '这个', '那个', '这样', '那样', '怎么', '因为', '所以',
        '但是', '可以', '知道', '觉得', '现在', '时候', '问题', '一下', '东西'
    ])
    combined_stopwords = ENGLISH_STOPWORDS.union(default_chinese_stopwords)
    if custom_stopwords:
        # Ensure custom stopwords are also lowercased if they might be English
        custom_lower = {sw.lower() for sw in custom_stopwords}
        combined_stopwords.update(custom_lower)

    # 3. Tokenization using Jieba
    # Precise mode (cut_all=False) is default and usually best
    tokens = jieba.cut(cleaned_text)

    # 4. Filter tokens
    words = []
    for token in tokens:
        word = token.strip()
        # Filter out: empty strings, stopwords, single characters (Chinese or English), and pure numbers (if any survived cleaning)
        if word and word not in combined_stopwords and len(word) > 1 and not word.isdigit():
            words.append(word)

    return Counter(words)

# --- End Text Processing Functions ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    """
    Receives raw text, processes it based on 'language' hint, generates word cloud.
    Handles 'en' and 'zh' (which implies mixed Chinese/English).
    Request Body (JSON):
    {
      "text": "Your large block of text here...",
      "language": "zh", // 'en' or 'zh' (processes mixed zh/en)
      "width": 800, // Optional
      "height": 600, // Optional
      "background_color": "white", // Optional
      "font_path": "/path/to/font.ttf", // Optional, overrides default. MUST SUPPORT CHINESE & ENGLISH
      "custom_stopwords": ["word1", "word2"], // Optional
      "max_words": 200 // Optional
    }
    Response (JSON):
    Success: {"imageUrl": "http://host:port/images/unique_id.png", "wordCount": 150}
    Failure: {"error": "Error message"}, status code 400 or 500
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    raw_text = data.get('text')
    # Treat 'zh' as the mode for mixed Chinese/English processing
    language = data.get('language', 'zh').lower()
    width = data.get('width', 800)
    height = data.get('height', 600)
    bg_color = data.get('background_color', 'white')
    font_path = data.get('font_path', DEFAULT_FONT_PATH)
    custom_stopwords = data.get('custom_stopwords', [])
    max_words = data.get('max_words', 200)

    if not raw_text or not isinstance(raw_text, str):
        return jsonify({"error": "Missing or invalid 'text' field in request body"}), 400

    # --- Font Path Validation ---
    final_font_path = font_path
    if not os.path.exists(final_font_path):
        logging.warning(f"Font path '{final_font_path}' provided or default not found.")
        if final_font_path != DEFAULT_FONT_PATH and os.path.exists(DEFAULT_FONT_PATH):
            logging.warning(f"Falling back to default font path: {DEFAULT_FONT_PATH}")
            final_font_path = DEFAULT_FONT_PATH
        # Check again after potential fallback
        if not os.path.exists(final_font_path):
             error_msg = f"CRITICAL: Valid font file supporting required characters not found at '{final_font_path}'. Cannot generate word cloud. Configure DEFAULT_FONT_PATH or provide a valid 'font_path'."
             logging.error(error_msg)
             return jsonify({"error": error_msg}), 500
    # --- End Font Path Validation ---

    try:
        # --- Perform Text Processing based on language hint ---
        logging.info(f"Processing text with language mode: {language}")
        if language == 'zh': # This now handles mixed content
            word_counts = process_text_mixed_chinese_english(raw_text, custom_stopwords)
        elif language == 'en':
             word_counts = process_text_nltk(raw_text, custom_stopwords)
        else:
            # Optionally add more specific language modes later if needed
            return jsonify({"error": f"Unsupported language mode: {language}. Use 'en' or 'zh' (for mixed Chinese/English)."}), 400

        if not word_counts:
            logging.warning("No words found after processing the text.")
            return jsonify({"error": "No processable words found in the provided text after filtering."}), 400

        frequencies = dict(word_counts)
        logging.info(f"Found {len(frequencies)} unique words after processing.")

        # --- Generate Word Cloud ---
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=bg_color,
            font_path=final_font_path, # CRITICAL: Font must support both Chinese and English glyphs
            max_words=max_words,
            # Consider adding scale for higher resolution: scale=2
        ).generate_from_frequencies(frequencies)

        # --- Save Image and Return URL ---
        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        wordcloud.to_file(output_path)
        logging.info(f"Word cloud image saved to: {output_path}")

        image_url = f"{request.host_url}images/{filename}"

        return jsonify({"imageUrl": image_url, "wordCount": len(frequencies)})

    except RuntimeError as e: # Catch specific errors like Jieba not installed
        logging.error(f"Runtime error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate word cloud: {str(e)}"}), 500

@app.route('/images/<filename>')
def get_image(filename):
    """Serves the generated images."""
    safe_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    if not safe_path.startswith(os.path.abspath(OUTPUT_DIR)):
        return "Forbidden", 403
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except FileNotFoundError:
        return "Image not found", 404

if __name__ == '__main__':
    if JIEBA_AVAILABLE:
        jieba.initialize()

    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=False in production