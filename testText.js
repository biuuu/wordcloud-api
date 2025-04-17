const axios = require('axios');

const PYTHON_SERVICE_URL = 'http://192.168.33.6:5000'; // Adjust if needed

/**
 * Calls the Python service to generate a word cloud from raw text.
 * @param {string} rawText - The raw text content.
 * @param {object} options - Optional parameters.
 * @param {string} [options.language='en'] - 'en' or 'zh'.
 * @param {number} [options.width=800] - Image width.
 * @param {number} [options.height=600] - Image height.
 * @param {string} [options.background_color='white'] - Background color.
 * @param {string} [options.font_path] - Specific font path to override server default.
 * @param {string[]} [options.custom_stopwords] - List of words to ignore.
 * @param {number} [options.max_words=200] - Max words in the cloud.
 * @returns {Promise<{imageUrl: string, wordCount: number}>} - Object containing image URL and word count.
 */
async function generateWordCloudViaPython(rawText, options = {}) {
    const endpoint = `${PYTHON_SERVICE_URL}/generate-wordcloud`;
    const payload = {
        text: rawText,
        ...options // Pass all options directly
    };

    console.log(`Sending request to ${endpoint} with text length: ${rawText.length}, options:`, JSON.stringify(options, null, 2));

    try {
        const response = await axios.post(endpoint, payload, {
            headers: { 'Content-Type': 'application/json' },
            // Add a timeout for potentially long processing
            timeout: 60000 // 60 seconds timeout (adjust as needed)
        });

        if (response.data && response.data.imageUrl) {
            console.log('Successfully received response:', response.data);
            return response.data; // Return the whole response object { imageUrl, wordCount }
        } else {
            throw new Error('Invalid response format from Python service');
        }
    } catch (error) {
        let errorMessage = 'Failed to generate word cloud via Python service.';
         if (error.code === 'ECONNABORTED') {
            errorMessage += ' Request timed out.';
            console.error(errorMessage);
        } else if (error.response) {
            console.error('Error response from Python service:', error.response.status, error.response.data);
            errorMessage += ` Status: ${error.response.status}, Message: ${JSON.stringify(error.response.data?.error || error.response.data)}`;
        } else if (error.request) {
            console.error('No response received from Python service:', error.request);
            errorMessage += ' No response received.';
        } else {
            console.error('Error setting up request to Python service:', error.message);
            errorMessage += ` Setup error: ${error.message}`;
        }
        throw new Error(errorMessage);
    }
}

// --- Usage Example ---
async function main() {
  const sampleChineseText = `
      自然语言处理（NLP）是人工智能（AI）和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；
      自然语言认知则是让电脑“懂”人类的语言。其涵盖了许多不同的技术，例如分词、词性标注、命名实体识别、
      情感分析、文本摘要和机器翻译等。TF-IDF是一种常用的关键词提取算法，用于评估一个词对于一个文件集或
      一个语料库中的其中一份文件的重要程度。Jieba库提供了方便的实现。词云图是数据可视化的好方法。
  `;
  const customWords = ["自然语言处理", "命名实体识别", "词性标注", "TF-IDF", "词云图"];

  try {
      console.log("\n--- Generating Word Cloud using TF-IDF ---");
      const result = await generateWordCloudViaPython(sampleChineseText, {
          language: 'zh',
          extraction_method: 'tfidf', // Specify TF-IDF method
          top_k: 100,                 // How many top keywords to extract
          user_dict_words: customWords,
          shape: 'ellipse',
          width: 800, height: 500,
          background_color: null, // Use a solid background for CQ image
          mode: 'RGBA', // RGBA for transparency support if needed, though white bg makes it less critical
          scale: 2,
          colormap: 'tab20',
          scale: 2
      });
      console.log(`Word Cloud URL: ${result.imageUrl}`);
      console.log(`Extraction Method Used: ${result.extraction_method_used}`);
      console.log(`Word Count: ${result.wordCount}`);

  } catch (error) {
      console.error('\n--- ERROR ---');
      console.error(error.message);
  }
}
main();