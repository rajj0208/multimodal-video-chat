import os
import json
import numpy as np
import faiss
import cohere
import google.generativeai as genai
from PIL import Image
import io
import base64
from io import BytesIO
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize clients
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

max_pixels = 1568*1568  # Max resolution for images

def resize_image(pil_image):
    """Resize too large images"""
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path):
    """Convert images to a base64 string"""
    pil_image = Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"
    
    resize_image(pil_image)
    
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")
    
    return img_data

def get_query_embedding(query):
    """Embed search query"""
    try:
        response = co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query],
        )
        return np.array(response.embeddings.float[0])
    except Exception as e:
        print(f"Query embedding error: {e}")
        return None

def load_indexes_and_metadata(video_dir):
    """Load FAISS indexes and metadata from video directory"""
    try:
        # Load metadata
        metadata_path = os.path.join(video_dir, "metadatas.json")
        with open(metadata_path, 'r') as f:
            metadatas = json.load(f)
        
        # Load FAISS indexes
        text_index_path = os.path.join(video_dir, "text_index.faiss")
        img_index_path = os.path.join(video_dir, "img_index.faiss")
        
        text_index = faiss.read_index(text_index_path)
        img_index = faiss.read_index(img_index_path)
        
        return text_index, img_index, metadatas
    except Exception as e:
        print(f"Error loading indexes and metadata: {e}")
        return None, None, None

def search_indexes(query_vector, text_index, img_index, k=15):
    """Search both text and image indexes"""
    try:
        # Search in text index
        text_distances, text_indices = text_index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        # Search in image index  
        img_distances, img_indices = img_index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        return text_distances[0], text_indices[0], img_distances[0], img_indices[0]
    except Exception as e:
        print(f"Error searching indexes: {e}")
        return None, None, None, None

def combine_results(text_distances, text_indices, img_distances, img_indices, text_weight=0.7, img_weight=0.3, min_k=2, max_k=10):
    """Combine text and image search results using an efficient scoring system"""
    # Convert distances to similarity scores
    text_similarities = 1 / (1 + text_distances)
    img_similarities = 1 / (1 + img_distances)
    
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Process text results
    for idx, sim in zip(text_indices, text_similarities):
        if idx not in combined_scores:
            combined_scores[idx] = {'text_score': sim, 'img_score': 0, 'count': 1}
        else:
            combined_scores[idx]['text_score'] = max(combined_scores[idx]['text_score'], sim)
            combined_scores[idx]['count'] += 1
    
    # Process image results
    for idx, sim in zip(img_indices, img_similarities):
        if idx not in combined_scores:
            combined_scores[idx] = {'text_score': 0, 'img_score': sim, 'count': 1}
        else:
            combined_scores[idx]['img_score'] = max(combined_scores[idx]['img_score'], sim)
            combined_scores[idx]['count'] += 1
    
    # Calculate final scores using weighted harmonic mean
    final_scores = []
    for idx, scores in combined_scores.items():
        # Use harmonic mean to favor frames that score well in both modalities
        if scores['text_score'] > 0 and scores['img_score'] > 0:
            harmonic_mean = 2 / (1/scores['text_score'] + 1/scores['img_score'])
        else:
            # If only one modality has a score, use that
            harmonic_mean = max(scores['text_score'], scores['img_score'])
        
        # Apply a bonus for frames that appear in both modalities
        if scores['count'] > 1:
            harmonic_mean *= 1.3  # 30% bonus for frames that appear in both modalities
        
        final_scores.append((idx, harmonic_mean))
    
    # Sort by score in descending order
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Find optimal k based on score distribution
    if not final_scores:
        return []
    
    # Calculate score differences between consecutive frames
    score_diffs = [final_scores[i][1] - final_scores[i+1][1] for i in range(len(final_scores)-1)]
    
    # Find the point of maximum score difference
    if score_diffs:
        max_diff_idx = np.argmax(score_diffs)
        optimal_k = max_diff_idx + 1
    else:
        optimal_k = 1
    
    # Ensure optimal_k is within bounds
    optimal_k = max(min_k, min(optimal_k, max_k))
    
    # If the top score is significantly higher than others, reduce k
    if len(final_scores) > 1 and final_scores[0][1] > 2 * final_scores[1][1]:
        optimal_k = max(min_k, optimal_k - 1)
    
    # Get the top k results
    top_results = []
    for idx, score in final_scores[:optimal_k]:
        # Find the corresponding distances
        text_dist = text_distances[np.where(text_indices == idx)[0][0]] if idx in text_indices else float('inf')
        img_dist = img_distances[np.where(img_indices == idx)[0][0]] if idx in img_indices else float('inf')
        
        top_results.append((idx, score, text_dist, img_dist))
    
    return top_results

def lvlm_inference(prompt, images, max_tokens=500, temperature=0.7):
    """Generate response using Gemini with images"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        # Prepare content
        content = [prompt]
        
        def base64_to_pil_image(base64_string):
            image_data = base64.b64decode(base64_string)
            return Image.open(BytesIO(image_data))
        
        # Convert all base64 images to PIL images
        if isinstance(images, list):
            pil_images = [base64_to_pil_image(img) for img in images]
            content.extend(pil_images)
        else:
            pil_image = base64_to_pil_image(images)
            content.append(pil_image)
        
        # Generate response
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        return response.text
        
    except Exception as e:
        raise RuntimeError(f"Error generating response with Gemini: {e}")

def video_search(query, video_dir, min_k=2, max_k=10, text_weight=0.7, img_weight=0.3):
    """
    Main function to search video content and get Gemini response with dynamic frame selection
    
    Args:
        query (str): Search query
        video_dir (str): Path to video directory containing indexes and metadata
        min_k (int): Minimum number of frames to return
        max_k (int): Maximum number of frames to return
        text_weight (float): Weight for text embeddings in combined results
        img_weight (float): Weight for image embeddings in combined results
    
    Returns:
        dict: Dictionary containing search results and Gemini response
    """
    print(f"Searching for: '{query}'")
    
    # Step 1: Create query embedding
    query_vector = get_query_embedding(query)
    if query_vector is None:
        return {"error": "Failed to create query embedding"}
    
    # Step 2: Load indexes and metadata
    text_index, img_index, metadatas = load_indexes_and_metadata(video_dir)
    if text_index is None or img_index is None or metadatas is None:
        return {"error": "Failed to load indexes or metadata"}
    
    # Step 3: Search indexes with larger initial k to allow for dynamic selection
    initial_k = max_k * 2  # Search more frames than we might need
    text_distances, text_indices, img_distances, img_indices = search_indexes(
        query_vector, text_index, img_index, k=15
    )
    if text_distances is None:
        return {"error": "Failed to search indexes"}
    
    # Step 4: Combine results with dynamic frame selection
    combined_results = combine_results(
        text_distances, text_indices, img_distances, img_indices,
        text_weight, img_weight, min_k, max_k
    )
    
    if not combined_results:
        return {"error": "No combined results found"}
    
    # Step 5: Prepare context and images for Gemini
    context_parts = []
    images_for_context = []
    
    for i, (idx, weighted_dist, text_dist, img_dist) in enumerate(combined_results):
        # Add transcript to context
        transcript = metadatas[idx]['transcript']
        context_parts.append(f"Context {i+1}: {transcript}")
        
        # Get image data for this result
        frame_path = metadatas[idx]['extracted_frame_path']
        try:
            image_base64_data = base64_from_image(frame_path)
            base64_string = image_base64_data.split(',')[1]
            images_for_context.append(base64_string)
        except Exception as e:
            print(f"Warning: Could not load image {frame_path}: {e}")
    
    # Step 6: Generate Gemini response
    if not images_for_context:
        return {
            "query": query,
            "text_results": [(idx, dist) for idx, dist in zip(text_indices, text_distances)],
            "image_results": [(idx, dist) for idx, dist in zip(img_indices, img_distances)],
            "top_results": combined_results,
            "metadatas": [metadatas[idx] for idx, _, _, _ in combined_results],
            "error": "No images found in combined results"
        }
    
    context_text = "\n\n".join(context_parts)
    prompt = f"""Based on the following context and images, please answer this question: "{query}"

Context from video transcripts:
{context_text}

Please provide a unified answer using both the textual context and the visual information from the images."""
    
    try:
        gemini_response = lvlm_inference(prompt, images_for_context, max_tokens=500, temperature=0.7)
        
        return {
            "query": query,
            "text_results": [(idx, dist) for idx, dist in zip(text_indices, text_distances)],
            "image_results": [(idx, dist) for idx, dist in zip(img_indices, img_distances)],
            "top_results": combined_results,
            "metadatas": [metadatas[idx] for idx, _, _, _ in combined_results],
            "gemini_response": gemini_response
        }
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return {
            "query": query,
            "text_results": [(idx, dist) for idx, dist in zip(text_indices, text_distances)],
            "image_results": [(idx, dist) for idx, dist in zip(img_indices, img_distances)],
            "top_results": combined_results,
            "metadatas": [metadatas[idx] for idx, _, _, _ in combined_results],
            "error": f"Failed to generate Gemini response: {e}"
        }

class VideoSearchConversation:
    """Class to handle conversational video search with context"""
    
    def __init__(self, video_dir, text_weight=0.7, img_weight=0.3):
        """Initialize conversation with video directory"""
        self.video_dir = video_dir
        self.text_weight = text_weight
        self.img_weight = img_weight
        self.conversation_history = []
        
        # Load indexes and metadata once
        self.text_index, self.img_index, self.metadatas = load_indexes_and_metadata(video_dir)
        if self.text_index is None or self.img_index is None or self.metadatas is None:
            raise ValueError("Failed to load indexes or metadata")
    
    def answer_from_context(self, query, max_tokens=500, temperature=0.7):
        """Answer a new query using Gemini with existing context"""
        if not hasattr(self, '_last_combined_results') or not self._last_combined_results:
            return {"error": "No previous search context available. Please perform a search first."}
        
        # Use the last search results
        combined_results = self._last_combined_results
        
        # Prepare context and images from last search
        context_parts = []
        images_for_context = []
        
        for i, (idx, weighted_dist, text_dist, img_dist) in enumerate(combined_results):
            transcript = self.metadatas[idx]['transcript']
            context_parts.append(f"Context {i+1}: {transcript}")
            
            try:
                frame_path = self.metadatas[idx]['extracted_frame_path']
                image_base64_data = base64_from_image(frame_path)
                base64_string = image_base64_data.split(',')[1]
                images_for_context.append(base64_string)
            except Exception as e:
                print(f"Warning: Could not load image {frame_path}: {e}")
        
        if not images_for_context:
            return {"error": "No images found in previous context"}
        
        # Build conversation context
        context_text = "\n\n".join(context_parts)
        
        if self.conversation_history:
            conversation_context = "Previous conversation:\n"
            for i, (prev_query, prev_response) in enumerate(self.conversation_history):
                conversation_context += f"Q{i+1}: {prev_query}\nA{i+1}: {prev_response}\n\n"
            
            prompt = f"""{conversation_context}You are a helpful video assistant. You have access to the same video frames and transcripts from the previous search. Answer the new question using this existing context as a conversational chatbot would. Your answer should be concise, relevant, and conversational—do not repeat the question or provide unnecessary explanations.

New question: {query}

Video transcript context (from previous search):
{context_text}

Use both the transcript and the video frames from the previous search to answer as if you are chatting with the user. If the answer is not clear from the provided context, politely say so."""
        else:
            prompt = f"""You are a helpful video assistant. You have access to video frames and transcripts. Answer the question using this context as a conversational chatbot would. Your answer should be concise, relevant, and conversational—do not repeat the question or provide unnecessary explanations.

Question: {query}

Video transcript context:
{context_text}

Use both the transcript and the video frames to answer as if you are chatting with the user. If the answer is not clear from the provided context, politely say so."""
        
        try:
            gemini_response = lvlm_inference(prompt, images_for_context, max_tokens=max_tokens, temperature=temperature)
            self.conversation_history.append((query, gemini_response))
            
            return {
                "query": query,
                "conversation_turn": len(self.conversation_history),
                "gemini_response": gemini_response,
                "used_previous_context": True,
                "context_sources": len(combined_results),
                "conversation_history": self.conversation_history.copy()
            }
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return {
                "query": query,
                "conversation_turn": len(self.conversation_history),
                "error": f"Failed to generate Gemini response: {e}"
            }

    def conversational_search(self, query, top_k=3):
        """Perform conversational search with context from previous queries"""
        # Step 1: Create query embedding
        query_vector = get_query_embedding(query)
        if query_vector is None:
            return {"error": "Failed to create query embedding"}
        
        # Step 2: Search indexes
        text_distances, text_indices, img_distances, img_indices = search_indexes(
            query_vector, self.text_index, self.img_index, k=5
        )
        if text_distances is None:
            return {"error": "Failed to search indexes"}
        
        # Step 3: Combine results
        combined_results = combine_results(
            text_distances, text_indices, img_distances, img_indices, 
            self.text_weight, self.img_weight, top_k
        )
        
        # Store results for potential reuse in answer_from_context
        self._last_combined_results = combined_results
        
        if not combined_results:
            return {"error": "No combined results found"}
        
        # Step 4: Prepare context and images for Gemini
        context_parts = []
        images_for_context = []
        
        for i, (idx, weighted_dist, text_dist, img_dist) in enumerate(combined_results):
            transcript = self.metadatas[idx]['transcript']
            context_parts.append(f"Context {i+1}: {transcript}")
            
            try:
                frame_path = self.metadatas[idx]['extracted_frame_path']
                image_base64_data = base64_from_image(frame_path)
                base64_string = image_base64_data.split(',')[1]
                images_for_context.append(base64_string)
            except Exception as e:
                print(f"Warning: Could not load image {frame_path}: {e}")
        
        if not images_for_context:
            return {
                "query": query,
                "conversation_turn": len(self.conversation_history),
                "top_results": combined_results,
                "metadatas": [self.metadatas[idx] for idx, _, _, _ in combined_results],
                "error": "No images found in combined results"
            }
        
        # Step 5: Build conversation context
        context_text = "\n\n".join(context_parts)
        
        if self.conversation_history:
            conversation_context = "Previous conversation:\n"
            for i, (prev_query, prev_response) in enumerate(self.conversation_history):
                conversation_context += f"Q{i+1}: {prev_query}\nA{i+1}: {prev_response}\n\n"
            
            prompt = f"""{conversation_context}You are a helpful video assistant. The following images are frames extracted from a video, and the transcripts are from the same video. Respond to the current question as a conversational chatbot would, using both the visual information from the video frames and the transcript context. Your answer should be concise, relevant, and conversational—do not repeat the question or provide unnecessary explanations.

Current question: {query}

Video transcript context:
{context_text}

Use both the transcript and the video frames to answer as if you are chatting with the user. If the answer is not clear from the provided context, politely say so."""
        else:
            prompt = f"""You are a helpful video assistant. The following images are frames extracted from a video, and the transcripts are from the same video. Respond to the user's question as a conversational chatbot would, using both the visual information from the video frames and the transcript context. Your answer should be concise, relevant, and conversational—do not repeat the question or provide unnecessary explanations.

Question: {query}

Video transcript context:
{context_text}

Use both the transcript and the video frames to answer as if you are chatting with the user. If the answer is not clear from the provided context, politely say so."""
        
        try:
            gemini_response = lvlm_inference(prompt, images_for_context, max_tokens=500, temperature=0.7)
            self.conversation_history.append((query, gemini_response))
            
            return {
                "query": query,
                "conversation_turn": len(self.conversation_history),
                "text_results": [(idx, dist) for idx, dist in zip(text_indices, text_distances)],
                "image_results": [(idx, dist) for idx, dist in zip(img_indices, img_distances)],
                "top_results": combined_results,
                "metadatas": [self.metadatas[idx] for idx, _, _, _ in combined_results],
                "gemini_response": gemini_response,
                "conversation_history": self.conversation_history.copy()
            }
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return {
                "query": query,
                "conversation_turn": len(self.conversation_history),
                "top_results": combined_results,
                "metadatas": [self.metadatas[idx] for idx, _, _, _ in combined_results],
                "error": f"Failed to generate Gemini response: {e}"
            }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        print("Conversation history reset.")
    
    def get_conversation_summary(self):
        """Get a summary of the conversation history"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = f"Conversation Summary ({len(self.conversation_history)} exchanges):\n"
        summary += "=" * 50 + "\n"
        for i, (query, response) in enumerate(self.conversation_history):
            summary += f"Q{i+1}: {query}\n"
            summary += f"A{i+1}: {response[:100]}...\n\n"
        return summary

def get_document_embedding(content, content_type="text"):
    """Embed document (text or image)"""
    try:
        if content_type == "text":
            response = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=[content],
            )
            return np.array(response.embeddings.float[0])
        else:
            api_input_document = {
                "content": [
                    {"type": "image", "image": base64_from_image(content)},
                ]
            }
            response = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
            return np.array(response.embeddings.float[0])
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def create_and_store_indexes(metadatas, index_dir):
    """
    Create embeddings from metadata and store FAISS indexes
    
    Args:
        metadatas (list): List of metadata dictionaries containing 'transcript' and 'extracted_frame_path'
        index_dir (str): Directory path to store the indexes
    
    Returns:
        tuple: (text_index, img_index) FAISS indexes or (None, None) if failed
    """
    try:
        import tqdm
        
        print("Creating embeddings...")
        text_embeddings = []
        img_embeddings = []

        for metadata in tqdm.tqdm(metadatas):
            # Get text embedding from transcript
            text_emb = get_document_embedding(metadata['transcript'], content_type="text")
            if text_emb is not None:
                text_embeddings.append(text_emb)
            
            # Get image embedding from extracted frame
            img_emb = get_document_embedding(metadata['extracted_frame_path'], content_type="image")
            if img_emb is not None:
                img_embeddings.append(img_emb)

        if not text_embeddings or not img_embeddings:
            print("Error: No embeddings were created")
            return None, None

        text_embeddings = np.vstack(text_embeddings)
        img_embeddings = np.vstack(img_embeddings)

        # Create FAISS indexes for both text and image embeddings
        dimension = text_embeddings.shape[1]  # Should be 1536

        # Create indexes
        text_index = faiss.IndexFlatIP(dimension)  
        img_index = faiss.IndexFlatIP(dimension)

        

        # Add embeddings to indexes
        text_index.add(text_embeddings.astype('float32'))
        img_index.add(img_embeddings.astype('float32'))

        print(f"Text index contains {text_index.ntotal} vectors")
        print(f"Image index contains {img_index.ntotal} vectors")

        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)

        # Save the FAISS indexes
        text_index_path = os.path.join(index_dir, "text_index.faiss")
        img_index_path = os.path.join(index_dir, "img_index.faiss")
        
        faiss.write_index(text_index, text_index_path)
        faiss.write_index(img_index, img_index_path)

        print(f"Text index saved to: {text_index_path}")
        print(f"Image index saved to: {img_index_path}")
        
        return text_index, img_index
        
    except Exception as e:
        print(f"Error creating and storing indexes: {e}")
        return None, None

# Example usage function
def example_usage():
    """Example of how to use the video_search function"""
    query = "Describe the earrings woman is wearing."
    video_dir = r"C:\Users\Raj244639\Documents\VideoChat\cohere-test\shared_data\videos\video5"
    
    result = video_search(query, video_dir, top_k=3)
    return result

if __name__ == "__main__":
    pass
    # Run conversational example when script is executed directly
    # conversational_example()