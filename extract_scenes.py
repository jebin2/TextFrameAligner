import os
import json
import pickle
from custom_logger import logger_config
from typing import List, Tuple, Dict, Optional
from scenedetect import detect, AdaptiveDetector
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from llm_scene_extract import run_transnetv2
from remove_duplicate import FaceDINO


# Global cache for embeddings and similarity models
EMBEDDING_CACHE = {}
SIMILARITY_MODELS = None


def variance_of_laplacian(image):
	"""Original method - Blur detection using Laplacian variance."""
	return cv2.Laplacian(image, cv2.CV_64F).var()


def sobel_variance(image):
	"""Sobel edge detection variance - often more robust than Laplacian."""
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
	sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
	return sobel_magnitude.var()


def tenengrad_sharpness(image):
	"""Tenengrad sharpness metric - very effective for natural images."""
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
	gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
	# Threshold to focus on significant edges
	threshold = gradient_magnitude.mean() + gradient_magnitude.std()
	return np.sum(gradient_magnitude[gradient_magnitude > threshold])


def brenner_sharpness(image):
	"""Brenner focus measure - good for high-frequency content."""
	# Horizontal gradient
	diff_h = np.abs(image[:, 2:] - image[:, :-2])
	# Vertical gradient  
	diff_v = np.abs(image[2:, :] - image[:-2, :])
	return np.sum(diff_h**2) + np.sum(diff_v**2)


def modified_laplacian_sharpness(image):
	"""Modified Laplacian - more robust than standard Laplacian."""
	# Use a larger kernel for better edge detection
	kernel = np.array([[-1, -1, -1],
					   [-1,  8, -1], 
					   [-1, -1, -1]], dtype=np.float32)
	
	laplacian = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
	return laplacian.var()


def edge_density_sharpness(image):
	"""Edge density based sharpness - good for complex scenes."""
	# Canny edge detection
	edges = cv2.Canny(image, 50, 150)
	edge_density = np.sum(edges) / edges.size
	
	# Combine with edge strength
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
	edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
	
	return edge_density * edge_strength


def gradient_magnitude_variance(image):
	"""Gradient magnitude variance - balanced approach."""
	# Scharr operator (more accurate than Sobel)
	scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
	scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
	gradient_magnitude = np.sqrt(scharrx**2 + scharry**2)
	return gradient_magnitude.var()


def local_contrast_sharpness(image):
	"""Local contrast based sharpness - good for overall image quality."""
	# Calculate local standard deviation using a sliding window
	kernel = np.ones((9, 9), np.float32) / 81
	local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
	local_sqr_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
	local_variance = local_sqr_mean - local_mean**2
	local_std = np.sqrt(np.maximum(local_variance, 0))
	return np.mean(local_std)


def wavelet_sharpness(image):
	"""Wavelet-based sharpness (requires scikit-image)."""
	try:
		from skimage import restoration
		# Simple wavelet-based approach using high-frequency content
		# Apply Gaussian blur and subtract from original
		blurred = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.0)
		high_freq = np.abs(image.astype(np.float32) - blurred)
		return np.mean(high_freq)
	except ImportError:
		# Fallback to high-pass filter
		kernel = np.array([[-1, -1, -1],
						  [-1,  9, -1],
						  [-1, -1, -1]], dtype=np.float32)
		high_pass = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
		return np.mean(np.abs(high_pass))


def composite_sharpness_score(image):
	"""
	Composite score combining multiple metrics for robust assessment.
	This is often the best approach for diverse content.
	"""
	# Normalize image to 0-255 range
	if image.dtype != np.uint8:
		image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	
	# Calculate multiple metrics
	laplacian_var = variance_of_laplacian(image)
	sobel_var = sobel_variance(image)
	tenengrad = tenengrad_sharpness(image)
	edge_density = edge_density_sharpness(image)
	local_contrast = local_contrast_sharpness(image)
	
	# Normalize each metric to 0-1 range (approximate)
	# These normalization factors might need adjustment based on your content
	laplacian_norm = min(laplacian_var / 1000, 1.0)
	sobel_norm = min(sobel_var / 5000, 1.0)
	tenengrad_norm = min(tenengrad / 1000000, 1.0)
	edge_norm = min(edge_density / 100, 1.0)
	contrast_norm = min(local_contrast / 50, 1.0)
	
	# Weighted combination (adjust weights based on your needs)
	weights = [0.2, 0.25, 0.25, 0.15, 0.15]
	composite_score = (weights[0] * laplacian_norm + 
					  weights[1] * sobel_norm +
					  weights[2] * tenengrad_norm +
					  weights[3] * edge_norm +
					  weights[4] * contrast_norm)
	
	return composite_score


def fast_sharpness_assessment(image):
	"""
	Fast sharpness assessment for real-time processing.
	Good balance between speed and accuracy.
	"""
	# Resize for faster processing if image is large
	h, w = image.shape[:2]
	if w > 640 or h > 480:
		scale = min(640/w, 480/h)
		new_w, new_h = int(w * scale), int(h * scale)
		image = cv2.resize(image, (new_w, new_h))
	
	# Use Scharr operator for better accuracy than Sobel
	scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
	scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
	gradient_magnitude = np.sqrt(scharrx**2 + scharry**2)
	
	# Focus on high-gradient regions
	threshold = np.percentile(gradient_magnitude, 75)
	high_gradient_mask = gradient_magnitude > threshold
	
	if np.sum(high_gradient_mask) > 0:
		return np.mean(gradient_magnitude[high_gradient_mask])
	else:
		return gradient_magnitude.mean()


def is_mostly_black(frame, black_threshold=20, percentage_threshold=0.9, sample_rate=10):
    """
    Fast black frame detection using pixel sampling.

    Args:
        frame: OpenCV BGR frame (NumPy array)
        black_threshold: grayscale value below which a pixel is considered black
        percentage_threshold: fraction of black pixels to consider frame mostly black
        sample_rate: sample every N-th pixel in both dimensions (higher = faster)
    Returns:
        True if mostly black, False otherwise
    """
    import cv2
    import numpy as np
    if frame is None or frame.size == 0:
        return True
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sample pixels
    sampled = gray[::sample_rate, ::sample_rate]
    black_count = np.sum(sampled < black_threshold)
    total_count = sampled.size
    return (black_count / total_count) >= percentage_threshold


def get_frame_hash(frame):
	"""Quick hash for frame identification."""
	# Simple hash based on mean and variance of frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (32, 32))  # Very small for fast hashing
	return hash((resized.mean(), resized.var()))


def load_embedding_cache(cache_path: str) -> Dict:
	"""Load embedding cache from disk."""
	cache_file = os.path.join(cache_path, "embedding_cache.pkl")
	if os.path.exists(cache_file):
		try:
			with open(cache_file, 'rb') as f:
				return pickle.load(f)
		except Exception as e:
			logger_config.warning(f"Failed to load embedding cache: {e}")
	return {}


def save_embedding_cache(cache: Dict, cache_path: str):
	"""Save embedding cache to disk."""
	os.makedirs(cache_path, exist_ok=True)
	cache_file = os.path.join(cache_path, "embedding_cache.pkl")
	try:
		with open(cache_file, 'wb') as f:
			pickle.dump(cache, f)
	except Exception as e:
		logger_config.warning(f"Failed to save embedding cache: {e}")


def get_image_embedding_cached(frame, model, processor, device, cache_path: str):
	"""Extract embedding with caching."""
	global EMBEDDING_CACHE
	
	# Create a hash for this frame
	frame_hash = get_frame_hash(frame)
	
	# Check cache first
	if frame_hash in EMBEDDING_CACHE:
		return EMBEDDING_CACHE[frame_hash]
	
	# Compute embedding
	embedding = get_image_embedding(frame, model, processor, device)
	
	# Cache it
	EMBEDDING_CACHE[frame_hash] = embedding
	
	return embedding


def get_image_embedding(frame, model, processor, device):
	"""Extract embedding from a frame using a vision transformer model."""
	import torch
	from PIL import Image
	
	# Convert BGR to RGB for PIL
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pil_image = Image.fromarray(rgb_frame)
	
	# Process image
	inputs = processor(images=pil_image, return_tensors="pt").to(device)
	
	# Extract embedding
	with torch.inference_mode():
		outputs = model(**inputs)
		# Use the [CLS] token embedding (first token of last hidden state)
		embedding = outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
	
	return embedding


def cosine_similarity_numpy(vec1, vec2):
	"""Compute cosine similarity between two vectors."""
	import numpy as np
	dot_product = np.dot(vec1, vec2)
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)
	
	if norm_vec1 == 0 or norm_vec2 == 0:
		return 0.0
	
	return dot_product / (norm_vec1 * norm_vec2)


def initialize_embedding_model():
	"""Initialize the vision transformer model for embeddings."""
	global SIMILARITY_MODELS
	
	# Return cached models if available
	if SIMILARITY_MODELS is not None:
		return SIMILARITY_MODELS
	
	import torch
	from transformers import AutoImageProcessor, AutoModel
	
	# Use a general-purpose vision model
	model_name = "google/vit-base-patch16-224-in21k"  # Pre-trained ViT model
	
	try:
		print("[INFO] Loading Vision Transformer model...")
		processor = AutoImageProcessor.from_pretrained(model_name)
		model = AutoModel.from_pretrained(model_name)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = model.to(device)
		model.eval()
		
		SIMILARITY_MODELS = (model, processor, device)
		return SIMILARITY_MODELS
	except Exception as e:
		print(f"Failed to load embedding model: {e}")
		print("Falling back to perceptual hashing...")
		SIMILARITY_MODELS = (None, None, None)
		return SIMILARITY_MODELS

def enhanced_black_detection(frame, black_threshold=15, percentage_threshold=0.85):
	"""Enhanced black frame detection with better thresholding."""
	if frame is None or frame.size == 0:
		return True
	
	# Convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Use Otsu's thresholding for adaptive threshold selection
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Calculate percentage of dark pixels
	dark_pixels = np.sum(binary < black_threshold)
	total_pixels = binary.size
	
	return (dark_pixels / total_pixels) >= percentage_threshold


def detect_freeze_frame(frame, previous_frames, threshold=0.95):
	"""Detect static/freeze frames by comparing with previous frames."""
	if len(previous_frames) < 3:
		return False
	
	# Compare with last few frames
	for prev_frame in previous_frames[-3:]:
		# Calculate structural similarity
		gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
		
		# Simple pixel difference method
		diff = cv2.absdiff(gray_current, gray_prev)
		similarity = 1.0 - (np.mean(diff) / 255.0)
		
		if similarity > threshold:
			return True
	return False


def detect_low_information(frame, entropy_threshold=4.0):
	"""Detect frames with low information content using entropy."""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Calculate histogram
	hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
	hist = hist.flatten()
	hist = hist[hist > 0]  # Remove zero entries
	
	# Calculate entropy
	prob = hist / hist.sum()
	entropy = -np.sum(prob * np.log2(prob))
	
	return entropy < entropy_threshold


def detect_text_heavy_frame(frame, text_ratio_threshold=0.3):
	"""Detect frames that are primarily text (often less meaningful for scene analysis)."""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Apply morphological operations to detect text-like structures
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	
	# Edge detection
	edges = cv2.Canny(morph, 50, 150)
	
	# Find contours that might be text
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	text_like_contours = 0
	for contour in contours:
		area = cv2.contourArea(contour)
		x, y, w, h = cv2.boundingRect(contour)
		aspect_ratio = w / h if h > 0 else 0
		
		# Text-like characteristics: small area, specific aspect ratio
		if 100 < area < 5000 and 0.1 < aspect_ratio < 10:
			text_like_contours += 1
	
	total_area = frame.shape[0] * frame.shape[1]
	text_ratio = text_like_contours / (total_area / 1000)  # Normalize
	
	return text_ratio > text_ratio_threshold


def is_meaningless_frame(frame):
	"""
	Comprehensive meaningless frame detection combining multiple methods.
	"""
	if frame is None or frame.size == 0:
		return True, "Empty frame"
	
	# 1. Enhanced black frame detection
	if enhanced_black_detection(frame):
		return True, "Black frame"
	
	# 2. Low sharpness detection (using your existing methods)
	sharpness = fast_sharpness_assessment(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	if sharpness < 10:  # Adjust threshold based on your needs
		return True, "Blurry frame"
	
	# 3. Low information content
	if detect_low_information(frame):
		return True, "Low information content"
	
	# 4. Text-heavy frames (optional, depending on use case)
	if detect_text_heavy_frame(frame):
		return True, "Text-heavy frame"
	
	return False, "Valid frame"

def resize_to_480p(frame):
    """
    Resize a frame to 480p max while keeping aspect ratio.
    If frame is already <= 480p in height, returns unchanged.
    """
    h, w = frame.shape[:2]
    if h <= 480:
        return frame  # already small enough
    
    scale = 480 / h
    new_w = int(w * scale)
    new_h = 480
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_sharpest_scene_frame(cap, scene_start: float, scene_end: float, fps: float, frames_dir: str, frame_index: int, dino: FaceDINO, sharpness_method='composite') -> Tuple[Optional[str], float, Optional[np.ndarray]]:
	"""
	Extracts the sharpest non-black frame within the scene using cached similarity detection.
	Returns (frame_path or None, best_timestamp, best_frame or None)
	"""
	start_frame = int(scene_start * fps)
	end_frame = int(scene_end * fps)
	step = max(1, (end_frame - start_frame) // 5)  # Sample max 5 frames

	best_var = -1
	best_frame = None
	best_time = scene_start

	# Select sharpness function
	sharpness_functions = {
		'laplacian': variance_of_laplacian,
		'sobel': sobel_variance,
		'tenengrad': tenengrad_sharpness,
		'brenner': brenner_sharpness,
		'modified_laplacian': modified_laplacian_sharpness,
		'edge_density': edge_density_sharpness,
		'gradient_variance': gradient_magnitude_variance,
		'local_contrast': local_contrast_sharpness,
		'wavelet': wavelet_sharpness,
		'composite': composite_sharpness_score,
		'fast': fast_sharpness_assessment
	}

	sharpness_func = sharpness_functions.get(sharpness_method, composite_sharpness_score)

	for f in range(start_frame, end_frame, step):
		cap.set(cv2.CAP_PROP_POS_FRAMES, f)
		ret, frame = cap.read()
		if not ret:
			continue

		dup, _ = dino.is_duplicate(frame)
		if dup:
			continue

		frame_480p = resize_to_480p(frame)
		# Check for black frames
		if is_mostly_black(frame_480p):
			continue

		gray = cv2.cvtColor(frame_480p, cv2.COLOR_BGR2GRAY)
		var = sharpness_func(gray)

		if var > best_var:
			best_var = var
			best_frame = frame.copy()
			best_time = f / fps

	if best_frame is not None:
		frame_filename = f"scene_{frame_index:04d}_at_{best_time:.2f}s.jpg"
		frame_path = os.path.join(frames_dir, frame_filename)
		cv2.imwrite(frame_path, best_frame)

		return frame_path, best_time, best_frame
	else:
		logger_config.warning(f"Failed to extract sharp non-black frame for scene {scene_start:.2f}-{scene_end:.2f}s")
		return None, best_time, None


def map_dialogues_to_scenes(scene_list: List[Tuple[float, float]], dialogues: List[dict],
							video_path: str, frames_dir: str, cache_path: str) -> List[dict]:
	"""
	Map each dialogue to its corresponding scene and save a sharp non-black frame.
	"""
	global EMBEDDING_CACHE
	
	os.makedirs(frames_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	# Load embedding cache
	EMBEDDING_CACHE = load_embedding_cache(cache_path)
	
	# Initialize embedding model (with caching)
	print("[INFO] Initializing similarity detection model...")
	model, processor, device = initialize_embedding_model()
	if model is not None:
		print("[INFO] Using Vision Transformer embeddings for similarity detection")
	else:
		print("[INFO] Using perceptual hashing for similarity detection")

	frames_extracted = 0
	scene_dialogue_map = []
	dino = FaceDINO(threshold=0.85)

	for i, (scene_start, scene_end) in tqdm(enumerate(scene_list), total=len(scene_list), desc="Processing scenes"):
		frame_path, best_time, best_frame = extract_sharpest_scene_frame(
			cap, scene_start, scene_end, fps, frames_dir, frames_extracted, dino
		)

		if frame_path:
			scene_dialogues = [
				d for d in dialogues
				if d['end'] >= scene_start and d['start'] <= scene_end
			]

			scene_dialogue_map.append({
				"scene_start": scene_start,
				"scene_end": scene_end,
				"best_time": best_time,
				"frame_path": [frame_path],
				"dialogues": scene_dialogues,
				"dialogue": ""
			})

			frames_extracted += 1

	cap.release()
	
	# Save embedding cache
	save_embedding_cache(EMBEDDING_CACHE, cache_path)
	
	return scene_dialogue_map


def combine_consecutive_same_dialogues(scene_map: List[dict]) -> List[dict]:
	if not scene_map:
		return []

	def are_dialogues_equal(d1, d2):
		return json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

	combined = []
	current = scene_map[0].copy()

	for next_scene in scene_map[1:]:
		next_dialogues = next_scene["dialogues"]
		current_dialogues = current["dialogues"]

		if are_dialogues_equal(current_dialogues, next_dialogues):
			current["scene_end"] = next_scene["scene_end"]
			current["frame_path"].extend(next_scene["frame_path"])
		else:
			combined.append(current)
			current = next_scene.copy()

	combined.append(current)
	return combined


def combine_dialogues(scene_map: List[dict]) -> List[dict]:
	if not scene_map:
		return []

	combined = []
	for current in scene_map:
		combined.append(current)
		combined[-1]["dialogue"] = " ".join([dia["text"] for dia in current["dialogues"]])

	return combined


def detect_scenes(video_path: str, frame_timestamps: List[float], threshold: float = 30.0) -> List[Tuple[float, float]]:
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()

	min_scene_len = 15
	scene_list = detect(
		video_path,
		AdaptiveDetector(min_scene_len=min_scene_len),
		show_progress=True
	)

	all_scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

	if frame_timestamps:
		matched_scenes = []
		for ts in frame_timestamps:
			closest_scene = min(all_scenes, key=lambda scene: min(abs(ts - scene[0]), abs(ts - scene[1])))
			matched_scenes.append(closest_scene)

		matched_scenes = list(set(matched_scenes))
	else:
		matched_scenes = all_scenes

	return matched_scenes


def extract_scenes(video_path: str, frame_timestamp: List[float], dialogues, cache_path, frames_dir, threshold: float = 30.0, start_from_sec = -1, end_from_sec = -1, skip_segment = [(None, None)]):
	try:
		cache_dir = f"{cache_path}/extract_scenes.json"

		if os.path.exists(cache_dir) and not frame_timestamp:
			logger_config.info(f"Using cached scene detection: {video_path}")
			with open(cache_dir, "r") as f:
				data = json.load(f)
			return data

		print("[INFO] Detecting scenes...")
		scenes = run_transnetv2(video_path, frame_timestamp, start_from_sec=start_from_sec, end_from_sec=end_from_sec, skip_segment=skip_segment)
		if len(scenes) == 0:
			raise ValueError("scenes is empty")

		print("[INFO] Mapping dialogues to scenes and extracting frames...")
		scene_dialogue_map = map_dialogues_to_scenes(scenes, dialogues, video_path, frames_dir, cache_path)

		if len(dialogues) > 0:
			print("[INFO] Combining consecutive scenes with identical dialogues...")
			# scene_dialogue_map = combine_consecutive_same_dialogues(scene_dialogue_map)
			scene_dialogue_map = combine_dialogues(scene_dialogue_map)

		with open(cache_dir, "w", encoding="utf-8") as f:
			json.dump(scene_dialogue_map, f, indent=2)
		print(f"\n[INFO] Mapping saved to {cache_dir}")

		return scene_dialogue_map
	except Exception as e:
		cleanup_models()
		raise e

def cleanup_models():
    """Clean up models and free memory."""
    global SIMILARITY_MODELS, EMBEDDING_CACHE
    
    if SIMILARITY_MODELS is not None:
        model, processor, device = SIMILARITY_MODELS
        
        if model is not None:
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        if processor is not None:
            del processor
        
        SIMILARITY_MODELS = None
    
    EMBEDDING_CACHE.clear()
    
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    print("[INFO] Models and cache cleaned up successfully")

if __name__ == "__main__":
	video_path = "../CaptionCreator/media/movie_review/The Reader 2008.mp4"
	cache_path = "temp2"
	os.makedirs(cache_path)
	frames_dir = f"{cache_path}/frames_dir"
	os.makedirs(frames_dir)
	extract_scenes(video_path, frame_timestamp=None, dialogues=None, cache_path=cache_path, frames_dir=frames_dir)