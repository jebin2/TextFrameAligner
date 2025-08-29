import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
import os
import time
import traceback

from vision_model import VisionModel

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class OpenGvLab(VisionModel):

    def load_model(self):
        pass

    def generate(self, image: Image.Image, text: str = "") -> str:

        model_name = "OpenGVLab/InternVL3-2B"
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,  # Use bfloat16 as recommended
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Load and preprocess image
        pixel_values = load_image(image).to(torch.bfloat16).cuda()
        
        # Add <image> token to prompt
        question = f'<image>\n{text}'
        
        # Generation config
        generation_config = dict(max_new_tokens=512, do_sample=False)
        
        # Generate response
        response = ""
        try:
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        finally:
            # Explicitly delete the large tensor and clear the cache
            del pixel_values
            torch.cuda.empty_cache()
            
        return response


if __name__ == "__main__":
    model = OpenGvLab()
    model.load_model()

    image_path = "/home/jebin/git/TextFrameAligner/temp_dir/Blue is the Warmest Color 2013_2161_4322/frames/scene_0000_at_9.38s.jpg"
    image = Image.open(image_path)

    text = (
        "Describe what is happening in this video frame as if you're telling a story. "
        "Focus on the main subjects, their actions, the setting, and any important details."
    )

    start_time = time.time()
    response = model.generate(image, text)
    end_time = time.time()

    print("OpenGvLab:", response)
    print(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")
