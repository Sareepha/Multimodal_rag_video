from yt_dlp import YoutubeDL
import base64
import cv2
import dataclasses
import glob
import json
import os
import PIL
import random
import requests
import textwrap
import time
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from enum import auto, Enum
from io import StringIO, BytesIO
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.prompt_values import PromptValue
from PIL import Image
from predictionguard import PredictionGuard
from pytubefix import YouTube, Stream
from tqdm import tqdm
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from urllib.request import urlopen
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
import numpy as np
MultimodalModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]]
import logging
from pathlib import Path
from typing import Optional
import dspy
import lancedb
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch


LANCEDB_HOST_FILE = "./shared_data/lancedb"

def get_latest_batch_data():
    """ดึง batch_id และ video_path ล่าสุดจาก LanceDB"""
    db = lancedb.connect(LANCEDB_HOST_FILE)
    table = db.open_table("test_tbl")
    records = table.to_pandas().to_dict('records')

    latest_batch_id = None
    latest_video_path = None
    if records:
        latest_batch_id = max(record['metadata']['batch_id'] for record in records if 'batch_id' in record['metadata'])
        for record in records:
            if record['metadata'].get('batch_id') == latest_batch_id:
                latest_video_path = record['metadata']['video_path']
                break

    if not latest_batch_id or not latest_video_path:
        raise ValueError("No valid batch_id or video_path found in the database.")

    return latest_batch_id, latest_video_path


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)

def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        return default

def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_prediction_guard_api_key():
    load_env()
    PREDICTION_GUARD_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY", None)
    if PREDICTION_GUARD_API_KEY is None:
        PREDICTION_GUARD_API_KEY = input("Please enter your Prediction Guard API Key: ")
    return PREDICTION_GUARD_API_KEY

PREDICTION_GUARD_URL_ENDPOINT = 'https://api.predictionguard.com'

templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]

def prepare_dataset_for_umap_visualization(hf_dataset, class_name, templates=templates, test_size=1000):
    dataset = load_dataset(hf_dataset, trust_remote_code=True)
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    test_dataset = train_test_dataset['test']
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)].format(class_name),
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs

def download_video(video_url, path='/tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

    try:
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

    except Exception as e:
        print(f"Error using pytube: {e}")
        print("Falling back to yt-dlp...")
        ydl_opts = {
            'outtmpl': os.path.join(path, '%(title)s.%(ext)s'),
            'format': 'mp4[height<=720]',
        }
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([video_url])
            if result == 0:
                filepath = glob.glob(os.path.join(path, '*.mp4'))
                return filepath[0] if len(filepath) > 0 else None
    return None

def get_video_id_from_url(video_url):
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]
    return video_url

# if this has transcript then download
def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path, 'captions.vtt')
    if os.path.exists(filepath):
        return filepath

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
        formatter = WebVTTFormatter()
        webvtt_formatted = formatter.format_transcript(transcript)

        with open(filepath, 'w', encoding='utf-8') as webvtt_file:
            webvtt_file.write(webvtt_formatted)
        webvtt_file.close()

        return filepath
    except Exception as e:
        print(f"Error downloading transcript: {e}")
        return None

def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"


def str2time(strtime):
    strtime = strtime.strip('"')
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds

def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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

def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def isBase64(sb):
    try:
        if isinstance(sb, str):
                sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
                sb_bytes = sb
        else:
                raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False


def encode_image_from_path_or_url(image_path_or_url):
    try:
        f = urlopen(image_path_or_url)
        return base64.b64encode(requests.get(image_path_or_url).content).decode('utf-8')
    except:
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

DEFAULT_IMAGE = Image.new('RGB', (224, 224), color=(255, 255, 255))

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        # ทำการ normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds[0].cpu().numpy()

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
        # ทำการ normalize
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds[0].cpu().numpy()

def clip_embedding(prompt, base64_image):
    """
    Compute the joint embedding of a prompt and a base64-encoded image using the CLIP model.

    Args:
        prompt (str): The text prompt to embed.
        base64_image (str): The base64-encoded image string.

    Returns:
        List[float]: The resulting embedding vector.
    """
    if base64_image:
        if not is_base64(base64_image):
            raise TypeError("Image input must be in base64 encoding!")
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise ValueError("Invalid image data!") from e
    else:
        raise ValueError("An image must be provided.")

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # คำนวณ embeddings ของภาพและข้อความ
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # Shape: [batch_size, embedding_dim]
        text_embeds = outputs.text_embeds  # Shape: [batch_size, embedding_dim]

        # ทำการ normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # รวม embeddings โดยการเฉลี่ย
        embedding_vector = (image_embeds + text_embeds) / 2
        embedding_vector = embedding_vector[0].cpu().numpy()

    return embedding_vector.tolist()



def load_json_file(file_path):
    # open the JSON file in read mode
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

from IPython.display import display
from PIL import Image

def display_retrieved_results(results):
    print(f'There is/are {len(results)} retrieved result(s)')
    print()
    for i, res in enumerate(results):
        # Display the caption
        print(f'The caption of the {str(i+1)}-th retrieved result is:\n{res.page_content}')
        print()

        # Check for the extracted frame path in metadata
        extracted_frame_path = res.metadata.get('extracted_frame_path')
        if not extracted_frame_path:
            print(f"Warning: No 'extracted_frame_path' found in metadata for result {i+1}.")
            print("------------------------------------------------------------")
            continue

        # Display the image inline in Jupyter
        try:
            img = Image.open(extracted_frame_path)
            display(img)  # Inline display in Jupyter
        except Exception as e:
            print(f"Error displaying image: {e}")
        print("------------------------------------------------------------")

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history"""
    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"

    def _get_prompt_role(self, role):
        if self.map_roles is not None and role in self.map_roles.keys():
            return self.map_roles[role]
        else:
            return role

    def _build_content_for_first_message_in_conversation(self, first_message: List[str]):
        content = []
        if len(first_message) != 2:
            raise TypeError("First message in Conversation needs to include a prompt and a base64-enconded image!")

        prompt, b64_image = first_message[0], first_message[1]

        # handling prompt
        if prompt is None:
            raise TypeError("API does not support None prompt yet")
        content.append({
            "type": "text",
            "text": prompt
        })
        if b64_image is None:
            raise TypeError("API does not support text only conversation yet")

        # handling image
        if not isBase64(b64_image):
            raise TypeError("Image in Conversation's first message must be stored under base64 encoding!")

        content.append({
            "type": "image_url",
            "image_url": {
                "url": b64_image,
            }
        })
        return content

    def _build_content_for_follow_up_messages_in_conversation(self, follow_up_message: List[str]):

        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError("Follow-up message in Conversation must not include an image!")

        # handling text prompt
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError("Follow-up message in Conversation must include exactly one text message")

        text = follow_up_message[0]
        return text

    def get_message(self):
        messages = self.messages
        api_messages = []
        for i, msg in enumerate(messages):
            role, message_content = msg
            if i == 0:
                # get content for very first message in conversation
                content = self._build_content_for_first_message_in_conversation(message_content)
            else:
                # get content for follow-up message in conversation
                content = self._build_content_for_follow_up_messages_in_conversation(message_content)

            api_messages.append({
                "role": role,
                "content": content,
            })
        return api_messages

    # this method helps represent a multi-turn chat into as a single turn chat format
    def serialize_messages(self):
        messages = self.messages
        ret = ""
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is not None and self.system != "":
                ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                role = self._get_prompt_role(role)
                if message:
                    if isinstance(message, List):
                        # get prompt only
                        message = message[0]
                    if i == 0:
                        # do not include role at the beginning
                        ret += message
                    else:
                        ret += role + ": " + message
                    if i < len(messages) - 1:
                        # avoid including sep at the end of serialized message
                        ret += self.sep
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        return ret

    def append_message(self, role, message):
        if len(self.messages) == 0:
            # data verification for the very first message
            assert role == self.roles[0], f"the very first message in conversation must be from role {self.roles[0]}"
            assert len(message) == 2, f"the very first message in conversation must include both prompt and an image"
            prompt, image = message[0], message[1]
            assert prompt is not None, f"prompt must be not None"
            assert isBase64(image), f"image must be under base64 encoding"
        else:
            # data verification for follow-up message
            assert role in self.roles, f"the follow-up message must be from one of the roles {self.roles}"
            assert len(message) == 1, f"the follow-up message must consist of one text message only, no image"
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x,y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }

prediction_guard_llava_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="Prediction Guard LLaVA enpoint Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={
        "user": "USER", 
        "assistant": "ASSISTANT"
    }
)

# get PredictionGuard client
def _getPredictionGuardClient():
    PREDICTION_GUARD_API_KEY = get_prediction_guard_api_key()
    client = PredictionGuard(
        api_key=PREDICTION_GUARD_API_KEY,
        url=PREDICTION_GUARD_URL_ENDPOINT,
    )
    return client


API_BASE = "your-api-base"
try:
    response = requests.get(API_BASE)
    if response.status_code == 200:
        print("API Server is running and accessible.")
    else:
        print(f"API Server returned status code {response.status_code}.")
except Exception as e:
    print(f"Error connecting to API Server: {e}")

# Setting up the LLaVA model via API Server
lm = dspy.LM(
    model="ollama/llava:34b",  
    api_base=API_BASE 
)

dspy.configure(lm=lm)
logging.info("LLaVA Model configured successfully.")

def lvlm_inference(prompt, image, max_tokens: int = 200, temperature: float = 0.3, top_p: float = 0.9, top_k: int = 50):
    # prepare conversation
    conversation = prediction_guard_llava_conv.copy()
    conversation.append_message(conversation.roles[0], [prompt, image])
    return lvlm_inference_with_conversation(conversation, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)

def lvlm_inference_with_conversation(conversation, max_tokens: int = 200, temperature: float = 0.3, top_p: float = 0.9, top_k: int = 50):
    try:
        messages = conversation.get_message()
        # Generate response by calling lm with messages
        response = lm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # ตรวจสอบว่า response เป็นลิสต์หรือไม่ 
        if isinstance(response, list):
            logging.warning("Response is a list, combining into a single string.")
            response = " ".join(response) 

        if not isinstance(response, str):
            raise ValueError("The response is not a string after processing.")

        return response

    except ValueError as ve:
        logging.error(f"Validation error during inference: {ve}")
        return {"error": str(ve)}

    except Exception as e:
        logging.error(f"Unexpected error during inference: {e}")
        return {"error": str(e)}


