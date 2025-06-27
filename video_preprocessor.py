from pathlib import Path
import os
from os import path as osp
import json
import cv2
from moviepy import VideoFileClip
from PIL import Image
import base64
from pytubefix import YouTube, Stream
from tqdm import tqdm
import glob
from io import StringIO, BytesIO
import dotenv
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

class VideoPreprocessor:
    """A class for downloading videos and extracting frames with AI-generated descriptions."""
    
    def __init__(self):
        self.lvlm_prompt = ("Describe all visual elements in this image including objects, people, text, colors, "
                           "spatial relationships, lighting, composition, setting, activities, emotions, clothing, "
                           "background details, and any other observable features. Focus on concrete details that "
                           "would enable accurate identification and retrieval of similar content.")
    
    def lvlm_inference(self, prompt, image, max_tokens: int = 200, temperature: float = 0.7):
        """Generate description for an image using Gemini AI."""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            content = [prompt]
            
            def base64_to_pil_image(base64_string):
                image_data = base64.b64decode(base64_string)
                return Image.open(BytesIO(image_data))
            
            pil_image = base64_to_pil_image(image)
            content.append(pil_image)
            
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
    
    def download_video(self, video_url, path='/tmp/'):
        """Download video from YouTube URL or return local file path."""
        print(f'Getting video information for {video_url}')
        if not video_url.startswith('http'):
            return os.path.join(path, video_url)

        filepath = glob.glob(os.path.join(path, '*.mp4'))
        if len(filepath) > 0:
            return filepath[0]

        def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
            pbar.update(len(data_chunk))
        
        yt = YouTube(video_url, on_progress_callback=progress_callback)
        stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
        if stream is None:
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        filepath = os.path.join(path, stream.default_filename)
        if not os.path.exists(filepath):   
            print('Downloading video from YouTube...')
            pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
            stream.download(path)
            pbar.close()
        
        return filepath
    
    def encode_image(self, image_path_or_PIL_img):
        """Encode image to base64 string."""
        if isinstance(image_path_or_PIL_img, Image.Image):
            buffered = BytesIO()
            image_path_or_PIL_img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            with open(image_path_or_PIL_img, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resize image while maintaining aspect ratio."""
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)
    
    def extract_and_save_frames_and_metadata_with_fps(self, path_to_video, path_to_save_extracted_frames, 
                                                     path_to_save_metadatas, num_of_extracted_frames_per_second=1):
        """Extract frames from video and generate metadata with AI descriptions."""
        metadatas = []
        video = cv2.VideoCapture(path_to_video)
        
        fps = video.get(cv2.CAP_PROP_FPS)
        hop = round(fps / num_of_extracted_frames_per_second) 
        curr_frame = 0
        idx = -1
        
        while True:
            ret, frame = video.read()
            if not ret: 
                break
            
            if curr_frame % hop == 0:
                idx = idx + 1
                image = self.maintain_aspect_ratio_resize(frame, height=350)
                
                img_fname = f'frame_{idx}.jpg'
                img_fpath = osp.join(path_to_save_extracted_frames, img_fname)
                cv2.imwrite(img_fpath, image)

                b64_image = self.encode_image(img_fpath)
                caption = self.lvlm_inference(self.lvlm_prompt, b64_image)
                    
                metadata = {
                    'extracted_frame_path': img_fpath,
                    'transcript': caption,
                    'video_segment_id': idx,
                    'video_path': path_to_video,
                }
                metadatas.append(metadata)
            
            curr_frame += 1
            
        metadatas_path = osp.join(path_to_save_metadatas, 'metadatas.json')
        with open(metadatas_path, 'w') as outfile:
            json.dump(metadatas, outfile)
        
        return metadatas
    
    def preprocess_video(self, video_url, output_directory, num_frames_per_second=1):
        """
        Download a video from URL and extract frames with metadata.
        
        Args:
            video_url (str): YouTube URL or local file path
            output_directory (str): Directory to save video and extracted content
            num_frames_per_second (int): Number of frames to extract per second
        
        Returns:
            list: Metadata for all extracted frames
        """
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        video_filepath = self.download_video(video_url, output_directory)
        
        extracted_frames_path = osp.join(output_directory, 'extracted_frames')
        metadatas_path = output_directory
        
        Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
        Path(metadatas_path).mkdir(parents=True, exist_ok=True)
        
        metadatas = self.extract_and_save_frames_and_metadata_with_fps(
            video_filepath,
            extracted_frames_path,
            metadatas_path,
            num_of_extracted_frames_per_second=num_frames_per_second
        )
        
        return metadatas