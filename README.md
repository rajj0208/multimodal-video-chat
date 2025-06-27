# 🎥 Multimodal Video Chat App

An AI-powered video analysis application that allows users to upload videos, extract frames, and have interactive conversations about video content using advanced multimodal AI models.

## 🌟 Features

- **Video Upload & Processing**: Upload videos and automatically extract frames with AI-generated descriptions
- **Multimodal Search**: Search through video content using both visual and textual information
- **Interactive Chat**: Have conversations about video content with AI assistance
- **Frame Visualization**: View and discuss specific video frames
- **Follow-up Questions**: Ask detailed questions about specific frames or scenes
- **Persistent Storage**: Save processed videos for future conversations
- **Beautiful UI**: Modern Streamlit interface with responsive design

## 🏗️ Architecture

The application uses a sophisticated multimodal architecture:

- **Video Processing**: Extracts frames at configurable intervals and generates AI descriptions
- **Dual Indexing**: Creates separate FAISS indexes for text (transcripts) and image embeddings
- **Semantic Search**: Combines text and image similarity search with weighted scoring
- **Conversational AI**: Maintains context across multiple conversation turns
- **Multimodal AI**: Uses Google Gemini for image understanding and Cohere for embeddings

## 🔧 Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **AI Models**: 
  - Google Gemini 2.0 Flash (vision and text generation)
  - Cohere Embed v4.0 (text and image embeddings)
- **Search**: FAISS (Facebook AI Similarity Search)
- **Video Processing**: OpenCV, MoviePy
- **Image Processing**: PIL, OpenCV
- **Environment**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Google Generative AI API key
- Cohere API key

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multimodal-video-chat-app.git
   cd multimodal-video-chat-app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   GENAI_API_KEY=your_google_generative_ai_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

## 🎯 Usage

### Starting the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload a Video**: 
   - Use the sidebar to upload a video file (MP4, AVI, MOV, MKV)
   - Set the frame extraction rate (frames per second)
   - Click "Process Video" to analyze the content

2. **Start Chatting**:
   - Select a processed video from the dropdown
   - Ask questions about the video content
   - View relevant frames that support the AI's responses

3. **Interactive Features**:
   - Ask follow-up questions about specific frames
   - Clear chat history to start fresh
   - View video statistics and sample frames

### Example Queries

- "What is the main character wearing?"
- "Describe the setting of this scene"
- "What objects can you see in the background?"
- "What emotions are visible in this video?"
- "Tell me about the lighting and composition"

## 📂 Project Structure

```
multimodal-video-chat-app/
├── streamlit_app.py           # Main Streamlit application
├── utils.py                   # Core utilities and search logic
├── video_preprocessor.py      # Video processing and frame extraction
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create this)
├── shared_data/
│   └── videos/               # Processed video data
│       └── [video_name]/
│           ├── [video_file]
│           ├── metadatas.json
│           ├── text_index.faiss
│           ├── img_index.faiss
│           └── extracted_frames/
│               └── frame_*.jpg
└── __pycache__/              # Python cache files
```

## 🔍 How It Works

### 1. Video Processing
- Videos are processed to extract frames at specified intervals
- Each frame is analyzed using Google Gemini to generate detailed descriptions
- Metadata including timestamps and descriptions is stored in JSON format

### 2. Dual Indexing System
- **Text Index**: Contains embeddings of frame descriptions and transcripts
- **Image Index**: Contains embeddings of actual video frames
- Both indexes use Cohere's embed-v4.0 model for high-quality embeddings

### 3. Multimodal Search
- User queries are embedded and searched against both indexes
- Results are combined using a weighted scoring system
- Optimal number of frames is selected based on score distribution

### 4. Conversational AI
- Maintains conversation history for context-aware responses
- Uses Google Gemini with both text and image inputs
- Supports follow-up questions on specific frames

## ⚙️ Configuration

### Frame Extraction Rate
- Adjustable from 0.5 to 3.0 frames per second
- Higher rates provide more detail but slower processing
- Recommended: 1.0 FPS for most use cases



## 📊 Performance

- **Processing Speed**: ~1-2 minutes per minute of video (1 FPS)
- **Memory Usage**: Scales with video length and frame rate
- **Search Speed**: Sub-second response times for most queries
- **Accuracy**: High relevance due to multimodal approach

## 🔒 Privacy & Security

- All processing happens locally or through secure API calls
- No video content is stored on external servers (except API calls)
- API keys are stored securely in environment variables
- Processed data is stored locally in the `shared_data` directory


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## 🙏 Acknowledgments

- **Google Generative AI** for powerful multimodal capabilities
- **Cohere** for high-quality embedding models
- **Facebook Research** for FAISS similarity search
- **Streamlit** for the excellent web framework

---

**Built with ❤️ using AI and modern technologies**
