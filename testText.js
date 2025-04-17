const axios = require('axios');

const PYTHON_SERVICE_URL = 'http://localhost:5000'; // Adjust if needed

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

    const sampleChineseText = `我今天使用了词云生成器，效果很棒。Node.js调用Python服务也很方便。`;

    const customWords = ["词云生成器", "Node.js"];

    try {
        // console.log("\n--- Generating English Word Cloud ---");
        // const resultEn = await generateWordCloudViaPython(sampleEnglishText, {
        //     language: 'en',
        //     width: 1000,
        //     height: 500,
        //     background_color: 'white',
        //     custom_stopwords: ['javascript', 'web'], // Add custom words to ignore
        // });
        // console.log(`English Word Cloud URL: ${resultEn.imageUrl} (Words: ${resultEn.wordCount})`);

        console.log("\n--- Generating Chinese Word Cloud ---");
        // IMPORTANT: Ensure the font path set in Python supports Chinese!
        const resultZh = await generateWordCloudViaPython(sampleChineseText, {
            language: 'zh',
            user_dict_words: customWords,
            shape: 'ellipse', // Specify the shape
            width: 800, height: 500, // Mask canvas size (rectangular for ellipse)
            background_color: null, // AliceBlue background (will fill ellipse)
            mode: 'RGBA', // Still use RGBA if you want clean edges
            scale: 2,
            colormap: 'tab20'
        });
        console.log(`Chinese Word Cloud URL: ${resultZh.imageUrl} (Words: ${resultZh.wordCount})`);

    } catch (error) {
        console.error('\n--- ERROR ---');
        console.error(error.message);
    }
}

main();