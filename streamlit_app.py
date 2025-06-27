import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import json
from video_preprocessor import VideoPreprocessor
from utils import VideoSearchConversation, create_and_store_indexes
import time

# Page configuration
st.set_page_config(
    page_title="Video Chat Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .video-info {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'current_video_dir' not in st.session_state:
        st.session_state.current_video_dir = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

def get_available_videos(base_dir="shared_data/videos"):
    """Get list of available processed videos"""
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            return []
        
        video_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has the required files
                metadata_file = os.path.join(item_path, "metadatas.json")
                text_index_file = os.path.join(item_path, "text_index.faiss")
                img_index_file = os.path.join(item_path, "img_index.faiss")
                
                if all(os.path.exists(f) for f in [metadata_file, text_index_file, img_index_file]):
                    video_dirs.append(item)
        
        return sorted(video_dirs)
    except Exception as e:
        st.error(f"Error loading available videos: {e}")
        return []

def process_new_video(video_file, video_name, frames_per_second=1):
    """Process a new video upload"""
    try:
        # Create directory for this video
        base_dir = "shared_data/videos"
        video_dir = os.path.join(base_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Create extracted_frames directory
        extracted_frames_dir = os.path.join(video_dir, "extracted_frames")
        os.makedirs(extracted_frames_dir, exist_ok=True)
        
        # Save uploaded video file
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        # Initialize preprocessor
        preprocessor = VideoPreprocessor()
        
        # Process video (extract frames and generate captions)
        st.info("🎬 Extracting frames and generating descriptions...")
        progress_bar = st.progress(0)
        
        metadatas = preprocessor.extract_and_save_frames_and_metadata_with_fps(
            video_path,
            extracted_frames_dir,
            video_dir,
            num_of_extracted_frames_per_second=frames_per_second
        )
        progress_bar.progress(50)
        
        # Create and store indexes
        st.info("🧠 Creating embeddings and building search indexes...")
        text_index, img_index = create_and_store_indexes(metadatas, video_dir)
        progress_bar.progress(100)
        
        if text_index is not None and img_index is not None:
            st.success(f"✅ Video '{video_name}' processed successfully!")
            st.info(f"📊 Created {len(metadatas)} frame descriptions and search indexes")
            return True
        else:
            st.error("❌ Failed to create search indexes")
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
        return False

def load_video_for_chat(video_dir_name):
    """Load a video for chatting"""
    try:
        video_dir = os.path.join("shared_data/videos", video_dir_name)
        
        # Initialize conversation
        conversation = VideoSearchConversation(video_dir)
        
        # Load metadata to show video info
        metadata_path = os.path.join(video_dir, "metadatas.json")
        with open(metadata_path, 'r') as f:
            metadatas = json.load(f)
        
        return conversation, metadatas
        
    except Exception as e:
        st.error(f"❌ Error loading video for chat: {e}")
        return None, None

def display_chat_history():
    """Display chat history with frame selection options"""
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        # Reverse the chat history to show most recent first
        for i, (query, response, frames_data) in enumerate(reversed(st.session_state.chat_history)):
            # Calculate the actual index for the original list (for unique keys)
            actual_index = len(st.session_state.chat_history) - 1 - i
            
            with st.container():
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {query}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong> {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Display frames if available
                if frames_data and len(frames_data) > 0:
                    with st.expander(f"🖼️ View frames from this response ({len(frames_data)} frames)"):
                        cols = st.columns(min(3, len(frames_data)))
                        for j, (frame_path, metadata) in enumerate(frames_data):
                            with cols[j % 3]:
                                try:
                                    from PIL import Image
                                    img = Image.open(frame_path)
                                    st.image(img, caption=f"Frame {metadata['video_segment_id']}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not load frame: {e}")
                        
                        # Add follow-up question option for this set of frames
                        follow_up_key = f"follow_up_{actual_index}"
                        follow_up_query = st.text_input(
                            f"Ask a follow-up question about these frames:",
                            key=follow_up_key,
                            placeholder="e.g., What else can you tell me about this scene?"
                        )
                        
                        if st.button(f"Ask Follow-up", key=f"btn_follow_up_{actual_index}") and follow_up_query.strip():
                            with st.spinner("Generating follow-up response..."):
                                try:
                                    # Use answer_from_context to respond using existing frames
                                    result = st.session_state.conversation.answer_from_context(
                                        follow_up_query.strip()
                                    )
                                    
                                    if 'gemini_response' in result:
                                        response = result['gemini_response']
                                        # Add to chat history with same frames
                                        st.session_state.chat_history.append((follow_up_query.strip(), response, frames_data))
                                        st.rerun()
                                    else:
                                        st.error("❌ Sorry, I couldn't generate a follow-up response.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                
                # Add a separator between chat messages
                st.markdown("---")

def main():
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">🎥 Video Chat Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for video selection and upload
    with st.sidebar:
        st.header("📹 Video Management")
        
        # Get available videos
        available_videos = get_available_videos()
        
        if available_videos:
            st.subheader("💾 Stored Videos")
            selected_video = st.selectbox(
                "Choose a video to chat with:",
                options=[""] + available_videos,
                format_func=lambda x: "Select a video..." if x == "" else x
            )
            
            if selected_video and selected_video != st.session_state.current_video_dir:
                with st.spinner("Loading video..."):
                    conversation, metadatas = load_video_for_chat(selected_video)
                    if conversation:
                        st.session_state.conversation = conversation
                        st.session_state.current_video_dir = selected_video
                        st.session_state.chat_history = []
                        st.success(f"✅ Loaded '{selected_video}'")
        else:
            st.info("No processed videos found. Upload a video below to get started!")
        
        st.markdown("---")
        
        # Video upload section
        st.subheader("📤 Upload New Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process and chat with"
        )
        
        if uploaded_file:
            video_name = st.text_input(
                "Video Name:",
                value=Path(uploaded_file.name).stem,
                help="Name for this video (will be used to save and identify it)"
            )
            
            frames_per_second = st.slider(
                "Frames per second to extract:",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Higher values extract more frames (more detailed but slower processing)"
            )
            
            if st.button("🚀 Process Video", type="primary"):
                if video_name.strip():
                    # Check if video name already exists
                    if video_name in available_videos:
                        st.error("❌ A video with this name already exists!")
                    else:
                        with st.spinner("Processing video... This may take several minutes."):
                            success = process_new_video(uploaded_file, video_name.strip(), frames_per_second)
                            if success:
                                st.rerun()
                else:
                    st.error("Please enter a video name!")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.conversation and st.session_state.current_video_dir:
            st.markdown(f"""
            <div class="video-info">
                <h3>🎬 Currently chatting with: {st.session_state.current_video_dir}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat input
            user_query = st.text_input(
                "Ask me anything about this video:",
                placeholder="e.g., What is happening in the video? Describe the main character...",
                key="user_input"
            )
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("💬 Send", type="primary") and user_query.strip():
                    with st.spinner("Thinking..."):
                        try:
                            result = st.session_state.conversation.conversational_search(
                                user_query.strip(), 
                                top_k=3
                            )
                            
                            if 'gemini_response' in result:
                                response = result['gemini_response']
                                
                                # Collect frame data for display
                                frames_data = []
                                if 'metadatas' in result and result['metadatas']:
                                    for metadata in result['metadatas']:
                                        frame_path = metadata['extracted_frame_path']
                                        frames_data.append((frame_path, metadata))
                                
                                st.session_state.chat_history.append((user_query.strip(), response, frames_data))
                                st.rerun()
                            else:
                                st.error("❌ Sorry, I couldn't generate a response. Please try again.")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            with col_clear:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    if st.session_state.conversation:
                        st.session_state.conversation.reset_conversation()
                    st.rerun()
            
            # Display chat history with frames
            display_chat_history()
            
        else:
            st.markdown("""
            <div class="chat-container">
                <h3>👋 Welcome to Video Chat Assistant!</h3>
                <p>To get started:</p>
                <ol>
                    <li><strong>Select an existing video</strong> from the sidebar if you have processed videos</li>
                    <li><strong>Or upload a new video</strong> using the upload section in the sidebar</li>
                    <li>Once a video is loaded, you can start chatting about its content!</li>
                </ol>
                <p>The AI will analyze both the visual content and any audio transcriptions to answer your questions.</p>
                <p><strong>New Feature:</strong> You can now ask follow-up questions about specific frames that appear in responses!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.current_video_dir:
            st.subheader("📊 Video Stats")
            try:
                metadata_path = os.path.join("shared_data/videos", st.session_state.current_video_dir, "metadatas.json")
                with open(metadata_path, 'r') as f:
                    metadatas = json.load(f)
                
                st.metric("Total Frames", len(metadatas))
                st.metric("Chat Turns", len(st.session_state.chat_history))
                
                # Show sample frame descriptions
                if metadatas:
                    st.subheader("🖼️ Sample Frames")
                    sample_size = min(3, len(metadatas))
                    for i, metadata in enumerate(metadatas[:sample_size]):
                        with st.expander(f"Frame {metadata['video_segment_id']}"):
                            # Display the frame image only
                            try:
                                from PIL import Image
                                img = Image.open(metadata['extracted_frame_path'])
                                st.image(img, caption=f"Frame {metadata['video_segment_id']}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not load frame: {e}")
                            
            except Exception as e:
                st.error(f"Error loading video stats: {e}")
        else:
            st.info("Select a video to see statistics and information.")

if __name__ == "__main__":
    main()