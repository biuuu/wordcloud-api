#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import logging
import math
import gc           # [优化] 引入垃圾回收模块
import ctypes       # [优化] 引入 C 类型接口，用于调用 malloc_trim
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from wordcloud import WordCloud
import numpy as np

# --- Text Processing Imports ---
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
    logging.info("Jieba library and analyse submodule found.")
except ImportError:
    JIEBA_AVAILABLE = False
    logging.critical("Jieba library or analyse submodule not found. Service cannot function.", exc_info=True)

# ================== CONFIGURATION ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_images')

# [配置] 字体路径 (请确保文件存在)
DEFAULT_FONT_PATH = './.local/HarmonyOS_SansSC_Regular.ttf'

STOPWORDS_FILENAME = 'baidu_stopwords.txt'
STOPWORDS_FILE_PATH = os.path.join(SCRIPT_DIR, STOPWORDS_FILENAME)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        import sys
        sys.exit("Exiting: Failed to create image output directory.")

# --- Load Chinese Stopwords At Startup ---
LOADED_CHINESE_STOPWORDS = set()
DEFAULT_HARDCODED_CHINESE_STOPWORDS = set(['的', '了', '和', '是', '就', '都', '而', '及', '在', '也'])
try:
    logging.info(f"Attempting to load Chinese stopwords from: {STOPWORDS_FILE_PATH}")
    with open(STOPWORDS_FILE_PATH, 'r', encoding='utf-8') as f:
        LOADED_CHINESE_STOPWORDS = {line.strip() for line in f if line.strip()}
    if not LOADED_CHINESE_STOPWORDS:
        LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except FileNotFoundError:
    logging.warning(f"Stopword file not found. Falling back to default list.")
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS
except Exception:
    LOADED_CHINESE_STOPWORDS = DEFAULT_HARDCODED_CHINESE_STOPWORDS

# ================== 核心优化功能 ==================

def force_memory_cleanup():
    """
    [核心优化] 强制执行全套内存回收流程
    专门针对 Linux 环境下的 Python 内存不释放问题
    """
    # 1. Python 层面的引用计数回收
    gc.collect()

    # 2. Linux 系统层面的堆内存修剪
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        # 非 Linux 系统可能会失败，直接忽略
        pass

# Initialize Flask application
app = Flask(__name__)

# --- Helper Function for Text Cleaning ---
def clean_chinese_text(text):
    if not isinstance(text, str): return ""
    return text

# --- Text Processing Function (Message TF-IDF) ---
def calculate_message_tfidf(raw_text_input, combined_stopwords, top_k=200):
    """
    计算 TF-IDF。
    [优化] 尽量减少中间变量的内存占用
    """
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required for TF-IDF.")
    if not raw_text_input: return {}

    # 使用 splitlines 获取列表，虽然这里产生了一次拷贝，但在 Flask 接收 JSON 时不可避免
    # 我们将在使用完后尽快释放 raw_text_input 的引用（在外部控制）
    lines = raw_text_input.splitlines()
    total_messages = len(lines)

    if total_messages == 0: return {}

    doc_freq = Counter()
    term_freq_total = Counter()

    logging.debug(f"Processing {total_messages} messages for TF-IDF...")

    # Pass 1: Tokenize
    for msg in lines:
        cleaned_msg = clean_chinese_text(msg)
        if not cleaned_msg: continue

        tokens = jieba.cut(cleaned_msg)

        # 过滤 (Generator expression to save memory)
        filtered_tokens = [
            word for word in (token.strip() for token in tokens)
            if word and word not in combined_stopwords and len(word) > 1
        ]

        if filtered_tokens:
            term_freq_total.update(filtered_tokens)
            doc_freq.update(set(filtered_tokens))

    # [优化] 处理完文本后，立刻尝试释放 lines 列表
    del lines

    if not term_freq_total: return {}

    # Pass 2: Calculate Scores
    tfidf_scores = {}

    # Pre-calculate log part to save CPU inside loop
    log_constant = math.log(total_messages + 1)

    for term, total_tf in term_freq_total.items():
        df = doc_freq.get(term, 0)
        # Simplify math
        idf = (log_constant - math.log(df + 1)) + 1.0
        tfidf_scores[term] = total_tf * idf

    # Sort and slice
    sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    # [优化] 释放中间的大字典
    del tfidf_scores
    del doc_freq
    del term_freq_total

    if not sorted_terms: return {}

    max_score = sorted_terms[0][1] if sorted_terms else 1.0
    scale_factor = 1000 / max_score if max_score > 0 else 1000

    final_frequencies = {
        word: max(1, int(score * scale_factor))
        for word, score in sorted_terms
    }

    return final_frequencies

# --- Text Processing Function (Global Frequency) ---
def process_text_chinese_frequency(text, combined_stopwords):
    if not JIEBA_AVAILABLE: raise RuntimeError("Jieba library is required.")

    tokens = jieba.cut(clean_chinese_text(text))
    # 使用生成器表达式进行过滤，避免产生中间的大列表
    words = (
        w for w in (t.strip() for t in tokens)
        if w and w not in combined_stopwords and len(w) > 1
    )
    return Counter(words)

# --- Mask Generation Functions ---
def create_circle_mask(width, height):
    radius = min(width, height) // 2; center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask_boolean = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image

def create_ellipse_mask(width, height):
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = max(1, width // 2), max(1, height // 2)
    y, x = np.ogrid[:height, :width]
    mask_boolean = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2) <= 1
    mask_image = np.full((height, width), 255, dtype=np.uint8); mask_image[mask_boolean] = 0
    return mask_image

# --- Flask Routes ---

@app.route('/generate-wordcloud', methods=['POST'])
def generate_wordcloud_route():
    # 初始化变量以便 finally 块清理
    wordcloud_instance = None
    frequencies = None
    mask_array = None

    try:
        if not JIEBA_AVAILABLE:
            return jsonify({"error": "Service Unavailable: Text library missing."}), 503

        # [优化] 1. 严格限制输入大小 (例如最大 5MB 文本)
        if request.content_length and request.content_length > 5 * 1024 * 1024:
            return jsonify({"error": "Payload too large. Max 5MB allowed."}), 413

        if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()

        raw_text = data.get('text')
        if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
            return jsonify({"error": "Missing 'text'"}), 400

        # [优化] 2. 限制图片分辨率和 Max Words，防止内存爆炸
        width = min(int(data.get('width', 800)), 3000)   # 强制最大宽 3000
        height = min(int(data.get('height', 600)), 3000) # 强制最大高 3000
        max_words = min(int(data.get('max_words', 200)), 1000) # 限制词数
        scale = min(float(data.get('scale', 1)), 2.0)    # 限制放大倍数

        custom_stopwords = data.get('custom_stopwords', [])
        user_dict_words = data.get('user_dict_words', [])
        weighting_scheme = data.get('weighting_scheme', 'message_tfidf').lower()
        shape = data.get('shape', 'rectangle').lower()
        font_path_req = data.get('font_path', None)
        mode = data.get('mode', 'RGBA')
        colormap_name = data.get('colormap', 'tab20')
        background_color = data.get('background_color', None)
        top_k = int(data.get('top_k', max_words))

        final_font_path = font_path_req if font_path_req else DEFAULT_FONT_PATH
        if not os.path.exists(final_font_path):
            return jsonify({"error": "Font file not found."}), 500

        # User Dict
        if user_dict_words:
            for word in user_dict_words:
                if isinstance(word, str) and word.strip():
                    jieba.add_word(word.strip())

        # Stopwords
        combined_stopwords = LOADED_CHINESE_STOPWORDS.copy()
        if custom_stopwords:
            combined_stopwords.update(sw for sw in custom_stopwords if isinstance(sw, str) and sw.strip())

        # Processing
        logging.info(f"Processing text with scheme: {weighting_scheme}...")

        if weighting_scheme == 'message_tfidf':
            frequencies = calculate_message_tfidf(raw_text, combined_stopwords, top_k)
        else:
            frequencies = dict(process_text_chinese_frequency(raw_text, combined_stopwords))

        if not frequencies:
            return jsonify({"error": "No valid words extracted."}), 400

        # Mask
        if shape != 'rectangle':
            try:
                if shape == 'circle': mask_array = create_circle_mask(width, height)
                elif shape == 'ellipse': mask_array = create_ellipse_mask(width, height)
            except Exception:
                shape = 'rectangle' # Fallback

        # Generation
        logging.info("Generating WordCloud image...")
        wordcloud_instance = WordCloud(
            width=width, height=height, background_color=background_color,
            font_path=final_font_path, max_words=max_words, mask=mask_array,
            mode=mode, scale=scale, colormap=colormap_name
        )
        wordcloud_instance.generate_from_frequencies(frequencies)

        # Save
        filename = f"wc_{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        wordcloud_instance.to_file(output_path)

        # [优化] 图片存盘后，立即清理 WordCloud 对象的大内存占用
        del wordcloud_instance
        wordcloud_instance = None # 防止 finally 再次删除报错

        image_url = f"{request.host_url}images/{filename}"

        return jsonify({
            "imageUrl": image_url,
            "success": True
        }), 200

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # [优化] 3. 请求结束时的终极清理
        # 显式删除当前作用域内可能残留的大对象
        if 'frequencies' in locals() and frequencies: del frequencies
        if 'mask_array' in locals() and mask_array is not None: del mask_array
        if 'wordcloud_instance' in locals() and wordcloud_instance: del wordcloud_instance
        if 'data' in locals(): del data
        if 'raw_text' in locals(): del raw_text

        # 强制调用系统回收
        force_memory_cleanup()
        logging.info("Memory cleanup executed.")

@app.route('/images/<path:filename>')
def get_image(filename):
    safe_dir = os.path.abspath(OUTPUT_DIR)
    try:
        return send_from_directory(safe_dir, filename, as_attachment=False)
    except FileNotFoundError:
        return "Image not found", 404

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "jieba": JIEBA_AVAILABLE}), 200

if __name__ == '__main__':
    if JIEBA_AVAILABLE:
        try:
            jieba.initialize()
        except: pass

    # 这里的 run 仅用于开发，生产环境请使用 Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)