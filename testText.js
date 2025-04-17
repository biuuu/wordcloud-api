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
    const sampleEnglishText = `
        Node.js is an open-source, cross-platform, back-end JavaScript runtime environment
        that runs on the V8 engine and executes JavaScript code outside a web browser.
        Node.js lets developers use JavaScript to write command line tools and for server-side
        scripting—running scripts server-side to produce dynamic web page content before the
        page is sent to the user's web browser. Consequently, Node.js represents a "JavaScript everywhere"
        paradigm, unifying web application development around a single programming language,
        rather than different languages for server-side and client-side scripts.
        Python is also very popular for backend development and data science. API design is important.
    `;

    const sampleChineseText = `
        词云（又称文字云或标签云）是文本数据的视觉表示。通常用于描述网站关键字（标签），
        或可视化自由格式文本。标签通常是单个词语，每个标签的重要性以字体大小或颜色显示。
        此格式有助于快速感知最突出的术语以确定其相对重要性。但较大的术语可能比较小的术语更难阅读。
        Python的wordcloud库是生成词云图的常用工具，需要配合jieba进行中文分词。
        设置正确的字体路径至关重要。微服务架构允许将不同功能模块部署为独立服务。
    `;

    try {
        console.log("\n--- Generating English Word Cloud ---");
        const resultEn = await generateWordCloudViaPython(sampleEnglishText, {
            language: 'en',
            width: 1000,
            height: 500,
            background_color: 'lightblue',
            custom_stopwords: ['javascript', 'web'], // Add custom words to ignore
            font_path: './.local/HarmonyOS_SansSC_Regular.ttf'
        });
        console.log(`English Word Cloud URL: ${resultEn.imageUrl} (Words: ${resultEn.wordCount})`);

        console.log("\n--- Generating Chinese Word Cloud ---");
        // IMPORTANT: Ensure the font path set in Python supports Chinese!
        const resultZh = await generateWordCloudViaPython(sampleChineseText, {
            language: 'zh',
            width: 900,
            height: 600,
            background_color: '#FFFFE0', // Light yellow
            font_path: './.local/HarmonyOS_SansSC_Regular.ttf' // Optionally override Python default here
        });
        console.log(`Chinese Word Cloud URL: ${resultZh.imageUrl} (Words: ${resultZh.wordCount})`);

    } catch (error) {
        console.error('\n--- ERROR ---');
        console.error(error.message);
    }
}

main();