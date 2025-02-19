from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from tqdm import tqdm
from typing import List
from utils import encode_image,clip_embedding
import PIL
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
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
    
class CLIPEmbeddings(BaseModel, Embeddings):
    """CLIP embedding model"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # เราสามารถคำนวณ embeddings ของข้อความได้
        embeddings = []
        for text in tqdm(texts, total=len(texts)):
            embedding = get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return get_text_embedding(text)

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        embeddings = []
        for path_to_img in tqdm(images, total=len(images)):
            image = Image.open(path_to_img).convert("RGB")
            embedding = get_image_embedding(image)
            embeddings.append(embedding)
        return embeddings
