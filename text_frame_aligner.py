import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger().setLevel(logging.ERROR)

from custom_logger import logger_config
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc
from gemiwrap import GeminiWrapper
from google import genai
import cv2
from scenedetect import detect, ContentDetector
import hashlib
import re
from pathlib import Path

TEMP_DIR = "temp_dir"
CACHE_DIR = f"{TEMP_DIR}/cache_dir"
OUTPUT_JSON = f'{TEMP_DIR}/output.json'

class TextFrameAligner:
	def __init__(self, blip_model_name="Salesforce/blip-image-captioning-large", sentence_model_name='all-mpnet-base-v2', clip_model_name="openai/clip-vit-large-patch14", max_workers=None):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.max_workers = max_workers or max(8, mp.cpu_count()-4)
		logger_config.info("TextFrameAligner initialization started")
		logger_config.info(f"Compute device: {self.device}, Max workers: {self.max_workers}")
		
		# Environment setup with optimizations
		os.environ["HF_HUB_TIMEOUT"] = "120"
		if torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
			torch.set_float32_matmul_precision('high')  # Use Tensor Cores on RTX cards

		self.blip_model_name = blip_model_name
		self.sentence_model_name = sentence_model_name
		self.clip_model_name = clip_model_name
		self.cache_path = None

		# Model containers
		self.processor = None
		self.blip_model = None
		self.clip_processor = None
		self.clip_model = None
		self.embedder = None

		# Cached data
		self.subtitles = []
		self.subtitle_embeddings = None
		self._sentence_cache = {}  # Cache for sentence embeddings
		self._clip_text_cache = {}  # Cache for CLIP text embeddings
		
		# weights
		self.weights = {
			'clip': 0.40,
			# 'semantic': 0.35,
			# 'tfidf': 0.15,  # Reduced as it's computationally expensive
			# 'subtitle': 0.25,
			# 'temporal': 0.05
		}
		
		logger_config.info("TextFrameAligner initialization completed")

	def set_cache_dir(self, identifier: str) -> str:
		"""Generates a unique cache directory path for a given identifier (e.g., video path or text)."""
		content_hash = hashlib.md5(identifier.encode()).hexdigest()
		cache_path = os.path.join(CACHE_DIR, content_hash)
		if not os.path.exists(cache_path):
			self.reset()
		os.makedirs(cache_path, exist_ok=True)
		self.cache_path = cache_path

	def load_clip_on_demand(self):
		"""Load CLIP only when needed and optimize for inference"""
		if self.clip_model is not None:
			return

		# Load CLIP
		from transformers import CLIPProcessor, CLIPModel
		self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
		self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

	@torch.inference_mode()
	def load_sentence_transformer(self):
		"""Load all models at once to avoid repeated loading/unloading"""
		logger_config.info("Loading SentenceTransformer")

		# Load SentenceTransformer
		self.embedder = SentenceTransformer(self.sentence_model_name)
		if self.device == "cuda":
			self.embedder = self.embedder.to(self.device)

		logger_config.info("SentenceTransformer loaded successfully")

	def load_blip_on_demand(self):
		"""Load BLIP only when needed and optimize for inference"""
		if self.blip_model is not None:
			return

		logger_config.info(f"Loading BLIP model: {self.blip_model_name}")
		from transformers import BlipProcessor, BlipForConditionalGeneration
		self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
		self.blip_model = BlipForConditionalGeneration.from_pretrained(
			self.blip_model_name,
			torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
		).to(self.device)
		
		# Optimize for inference
		if self.device == "cuda":
			self.blip_model = torch.compile(self.blip_model, mode="reduce-overhead")

	def unload_blip(self):
		if self.blip_model:
			logger_config.info("Unloading BLIP model")
			del self.blip_model, self.processor
			self.blip_model = None
			self.processor = None
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def extract_scenes(self, video_path: str, threshold: float = 30.0) -> Tuple[List[str], List[int], List[float]]:
		"""Optimized scene extraction with faster detection and efficient frame extraction."""
		cache_dir = f"{self.cache_path}/extract_scenes.json"
		if os.path.exists(cache_dir):
			logger_config.info(f"Using cache scene detection: {video_path}")
			with open(cache_dir, "r") as f:
				data = json.load(f)
			return data["frame_paths"], data["frame_numbers"], data["timestamps"]

		logger_config.info(f"Starting optimized scene detection: {video_path}")

		# 1. OPTIMIZATION: Use faster scene detection with downscaling and frame skipping
		scene_list = detect(
			video_path, 
			ContentDetector(threshold=threshold),
			show_progress=True
		)
		
		if not scene_list:
			logger_config.warning("No scenes found.")
			return [], [], []
		
		logger_config.info(f"Found {len(scene_list)} scenes")

		# 2. OPTIMIZATION: Pre-open video and get properties once
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise IOError(f"Cannot open video file: {video_path}")
		
		# Get video properties once
		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		if fps <= 0:
			cap.release()
			raise ValueError("Could not read FPS from video.")

		frames_dir = os.path.join(TEMP_DIR, "frames")
		os.makedirs(frames_dir, exist_ok=True)

		# 3. OPTIMIZATION: Create sorted frame extraction plan
		extraction_plan = []
		for i, (start_time, _) in enumerate(scene_list):
			start_frame = min(start_time.get_frames(), total_frames - 1)  # Boundary check
			timestamp = start_time.get_seconds()
			extraction_plan.append((start_frame, i, timestamp))
		
		# Sort by frame number for sequential reading
		extraction_plan.sort(key=lambda x: x[0])
		
		frame_paths, frame_numbers, timestamps = [], [], []
		
		# 4. OPTIMIZATION: Sequential frame reading with smart seeking
		logger_config.info(f"Starting optimized frame extraction for {len(extraction_plan)} frames...")
		
		current_frame_pos = 0
		frames_extracted = 0
		
		for target_frame, scene_index, timestamp in extraction_plan:
			# Smart seeking: only seek if we're far from target
			if abs(target_frame - current_frame_pos) > 10:
				cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
				current_frame_pos = target_frame
			else:
				# Skip frames if we're close to target
				while current_frame_pos < target_frame:
					cap.read()
					current_frame_pos += 1
			
			# Read the target frame
			ret, frame = cap.read()
			if not ret:
				logger_config.warning(f"Could not read frame {target_frame}")
				continue
			
			# 5. OPTIMIZATION: Immediate processing without thread overhead for small batches
			try:
				with Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) as image:
					# Quick black frame check with reduced resolution
					if self.is_mostly_black_fast(image):
						logger_config.info(f"Skipping black frame {target_frame}")
						continue

					filename = f"scene_{scene_index:03d}_frame_{target_frame}_at_frame_second{timestamp:.2f}frame_second.jpg"
					frame_path = os.path.join(frames_dir, filename)

					# OPTIMIZATION: Faster image saving
					image.save(frame_path, format='JPEG', quality=80, optimize=False)
					
					frame_paths.append(frame_path)
					frame_numbers.append(target_frame)
					timestamps.append(timestamp)
					frames_extracted += 1
					
					logger_config.info(f"Extracted frame {frames_extracted}/{len(extraction_plan)}", overwrite=True)
			
			except Exception as e:
				logger_config.error(f"Error processing frame {target_frame}: {e}")
				continue
			
			current_frame_pos += 1
		
		cap.release()
		
		logger_config.info(f"Extracted {len(frame_paths)} frames using optimized method.")
		
		# Cache results
		with open(cache_dir, "w") as f:
			json.dump({
				"frame_paths": frame_paths,
				"frame_numbers": frame_numbers,
				"timestamps": timestamps
			}, f, indent=4)

		return frame_paths, frame_numbers, timestamps

	def is_mostly_black_fast(self, image: Image.Image, black_threshold=20, percentage_threshold=0.9):
		"""
		Faster black frame detection using downsampling.
		"""
		# OPTIMIZATION: Downsample image for faster processing
		width, height = image.size
		if width > 200 or height > 200:
			# Resize to max 200px for faster processing
			scale = min(200/width, 200/height)
			new_size = (int(width * scale), int(height * scale))
			image = image.resize(new_size, Image.LANCZOS)
		
		# Convert to grayscale
		grayscale_image = image.convert('L')
		
		# OPTIMIZATION: Use numpy for faster computation
		import numpy as np
		pixels = np.array(grayscale_image)
		
		# Count black pixels using vectorized operations
		black_pixel_count = np.sum(pixels < black_threshold)
		total_pixels = pixels.size
		
		black_percentage = black_pixel_count / total_pixels
		return black_percentage >= percentage_threshold

	@torch.inference_mode()
	def compute_frame_clip_embeddings(self, frame_paths: List[str]) -> torch.Tensor:
		"""Highly CLIP embedding computation with memory management"""
		cache_dir = f"{self.cache_path}/compute_frame_clip_embeddings.pt"
		if os.path.exists(cache_dir):
			logger_config.info("Using cache CLIP image embeddings")
			return torch.load(cache_dir)

		logger_config.info("Computing CLIP image embeddings")

		# Determine optimal batch size based on available VRAM
		if self.device == "cuda":
			free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
			batch_size = min(32, max(8, int(free_memory / (512 * 1024 * 1024))))  # Estimate based on memory
		else:
			batch_size = 8

		logger_config.info(f"Using batch size: {batch_size}")

		all_embeddings = []

		def load_and_preprocess_batch(paths):
			"""Image loading and preprocessing"""
			images = []
			for path in paths:
				try:
					img = Image.open(path)
					if img.mode != 'RGB':
						img = img.convert('RGB')
					images.append(img)
				except Exception as e:
					logger_config.warning(f"Failed to load {path}: {e}")
					continue
			return images

		for i in range(0, len(frame_paths), batch_size):
			batch_paths = frame_paths[i:i + batch_size]

			# Load images in parallel
			with ThreadPoolExecutor(max_workers=min(4, len(batch_paths))) as executor:
				images = load_and_preprocess_batch(batch_paths)

			if not images:
				continue

			# Process with CLIP
			inputs = self.clip_processor(
				text=None, images=images, return_tensors="pt", padding=True
			).to(self.device)

			# Use mixed precision for faster computation
			with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32):
				image_features = self.clip_model.get_image_features(inputs.pixel_values)
				image_features = image_features.float()  # Convert back to float32 for compatibility

			all_embeddings.append(image_features.cpu())

			# Clean up
			for img in images:
				img.close()
			del inputs, image_features

			if torch.cuda.is_available():
				torch.cuda.empty_cache()

			logger_config.debug(f"Processed batch {i//batch_size + 1}/{(len(frame_paths)-1)//batch_size + 1}", overwrite=True)

		frame_clip_embeddings = torch.cat(all_embeddings, dim=0)
		torch.save(frame_clip_embeddings, cache_dir)
		return frame_clip_embeddings

	@lru_cache(maxsize=128)
	def get_cached_sentence_embeddings(self, text: str) -> torch.Tensor:
		"""Cache sentence embeddings to avoid recomputation"""
		return self.embedder.encode([text], convert_to_tensor=True)

	@lru_cache(maxsize=128)
	def get_cached_clip_text_embeddings(self, text: str) -> torch.Tensor:
		"""Cache CLIP text embeddings"""
		inputs = self.clip_processor(text=[text], images=None, return_tensors="pt", padding=True).to(self.device)
		with torch.inference_mode():
			text_features = self.clip_model.get_text_features(**inputs)
		return text_features.cpu()

	def caption_generation(self, frame_paths: List[str], timestamps: List[float]) -> List[str]:
		"""Captioning with batching and mixed precision"""
		cache_dir = f"{self.cache_path}/caption_generation.json"
		if os.path.exists(cache_dir):
			logger_config.info(f"Using cache caption generation for {len(frame_paths)} frames")
			with open(cache_dir, "r") as f:
				return json.load(f)

		self.load_blip_on_demand()
		logger_config.info(f"Starting caption generation for {len(frame_paths)} frames")
		
		batch_size = 4  # Smaller batch for BLIP to manage memory
		captions = []
		
		for i in range(0, len(frame_paths), batch_size):
			batch_paths = frame_paths[i:i + batch_size]
			batch_timestamps = timestamps[i:i + batch_size]
			batch_images = []

			# Load batch images
			for path in batch_paths:
				try:
					img = Image.open(path)
					if img.mode != 'RGB':
						img = img.convert('RGB')
					batch_images.append(img)
				except Exception as e:
					logger_config.warning(f"Failed to load {path}: {e}")
					# Use placeholder
					batch_images.append(Image.new('RGB', (224, 224), color='black'))

			# Get subtitle context for batch
			batch_contexts = []
			for timestamp in batch_timestamps:
				relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=1.0)
				context = " ".join([sub['text'] for sub in relevant_subs[:2]])  # Limit context
				batch_contexts.append(context[:100] if context else "")  # Truncate

			# Process batch
			batch_captions = []
			for img, context in zip(batch_images, batch_contexts):
				try:
					if context.strip():
						inputs = self.processor(
							images=img,
							text=f"A scene where {context}",
							return_tensors="pt"
						).to(self.device)
					else:
						inputs = self.processor(images=img, return_tensors="pt").to(self.device)

					# Use mixed precision and generation
					with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32):
						out = self.blip_model.generate(
							**inputs,
							max_length=80,  # Reduced for speed
							num_beams=4,	# Reduced beams
							length_penalty=0.8,
							repetition_penalty=1.2,
							do_sample=False,  # Deterministic for speed
							early_stopping=True
						)
					
					caption = self.processor.decode(out[0], skip_special_tokens=True)
					if context.strip():
						caption = caption.replace(f"A scene where {context}", "").strip()
					batch_captions.append(caption)

				except Exception as e:
					logger_config.warning(f"Caption generation failed: {e}")
					batch_captions.append("A scene from the video")

				img.close()

			captions.extend(batch_captions)

			# Progress logging
			progress = min(100, (i + batch_size) / len(frame_paths) * 100)
			logger_config.info(f"Caption progress: {progress:.1f}%", overwrite=True)

		self.unload_blip()
		logger_config.info(f"Caption generation completed for {len(captions)} frames")
		with open(cache_dir, 'w') as f:
			json.dump(captions, f, indent=4)
		return captions

	def similarity_computation(self, captions: List[str], query: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
		"""Compute all similarity metrics efficiently in one pass"""
		
		# 1. Semantic similarity with caching
		if query not in self._sentence_cache:
			self._sentence_cache[query] = self.embedder.encode([query], convert_to_tensor=True)
		query_emb = self._sentence_cache[query]
		
		# Batch encode captions
		caption_embs = self.embedder.encode(captions, convert_to_tensor=True, batch_size=32)
		semantic_similarities = util.cos_sim(query_emb, caption_embs)[0]
		
		# 2. CLIP similarity with caching
		if query not in self._clip_text_cache:
			self._clip_text_cache[query] = self.get_cached_clip_text_embeddings(query)
		
		# 3. TF-IDF (only if we have enough captions to make it worthwhile)
		tfidf_similarities = np.zeros(len(captions))
		if len(captions) > 5:
			try:
				from sklearn.feature_extraction.text import TfidfVectorizer
				from sklearn.metrics.pairwise import cosine_similarity
				vectorizer = TfidfVectorizer(
					stop_words='english',
					ngram_range=(1, 2),
					# max_features=1000,  # Limit features for speed
					dtype=np.float32	 # Use float32 for speed
				)
				tfidf_matrix = vectorizer.fit_transform(captions + [query])
				tfidf_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
			except Exception as e:
				logger_config.warning(f"TF-IDF computation failed: {e}")
		
		return semantic_similarities, self._clip_text_cache[query], tfidf_similarities

	def load_subtitles(self, timestamp_data):
		"""subtitle loading with pre-computed embeddings"""
		logger_config.info(f"Loading timestamp_data")

		# Pre-compute embeddings in batch
		if timestamp_data:
			self.subtitles = timestamp_data
			subtitle_texts = [sub['text'] for sub in self.subtitles]
			self.subtitle_embeddings = self.embedder.encode(
				subtitle_texts,
				convert_to_tensor=True,
				batch_size=64  # Larger batch for faster processing
			)
			logger_config.info(f"Pre-computed embeddings for {len(subtitle_texts)} subtitles")

	def get_subtitles_for_timerange(self, start_time: float, end_time: float, buffer: float = 2.0) -> List[Dict]:
		"""subtitle retrieval with binary search"""
		if not self.subtitles:
			return []
		
		# Simple linear search is often faster for small datasets
		# Binary search would be beneficial for very large subtitle files
		relevant_subs = []
		for sub in self.subtitles:
			if sub['end'] >= start_time - buffer and sub['start'] <= end_time + buffer:
				relevant_subs.append(sub)
		
		return relevant_subs

	def find_best_match(self, captions: List[str], timestamps: List[float], query: str, frame_clip_embeddings: torch.Tensor, previous_timestamp: Optional[float] = None, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
		"""Highly matching with vectorized operations"""

		# Compute all similarities efficiently
		semantic_sims, query_clip_emb, tfidf_sims = self.similarity_computation(captions, query)

		# CLIP similarity (vectorized)
		clip_similarities = util.cos_sim(query_clip_emb, frame_clip_embeddings)[0]

		# Vectorized subtitle similarity computation
		subtitle_scores = np.zeros(len(timestamps))
		if self.subtitles and self.subtitle_embeddings is not None:
			query_sub_emb = self.get_cached_sentence_embeddings(query)
			for i, timestamp in enumerate(timestamps):
				relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=3.0)
				if relevant_subs:
					# Get indices and compute similarity
					indices = [self.subtitles.index(sub) for sub in relevant_subs if sub in self.subtitles]
					if indices:
						relevant_embeddings = self.subtitle_embeddings[indices]
						similarities = util.cos_sim(query_sub_emb, relevant_embeddings)[0]
						subtitle_scores[i] = float(torch.max(similarities))

		# Vectorized temporal coherence
		temporal_scores = np.zeros(len(timestamps))
		if previous_timestamp is not None:
			time_diffs = np.array(timestamps) - previous_timestamp
			# Forward progression bonus
			forward_mask = time_diffs > 0
			temporal_scores[forward_mask] = np.maximum(0, 1.0 - (time_diffs[forward_mask] / 30.0))
			# Backward penalty
			backward_mask = time_diffs <= 0
			temporal_scores[backward_mask] = -0.5 * np.minimum(1.0, np.abs(time_diffs[backward_mask]) / 10.0)

		# Vectorized final score computation
		semantic_weights = self.weights['semantic'] * semantic_sims.cpu().numpy()
		clip_weights = self.weights['clip'] * clip_similarities.cpu().numpy()
		tfidf_weights = self.weights['tfidf'] * tfidf_sims
		subtitle_weights = self.weights['subtitle'] * subtitle_scores
		temporal_weights = self.weights['temporal'] * temporal_scores

		final_scores = semantic_weights + clip_weights + tfidf_weights + subtitle_weights + temporal_weights

		# Get top-k indices efficiently
		top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
		top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

		# Build results
		results = []
		for idx in top_indices:
			score_breakdown = {
				'semantic': float(semantic_sims[idx]),
				'clip': float(clip_similarities[idx]),
				'tfidf': float(tfidf_sims[idx]),
				'subtitle': subtitle_scores[idx],
				'temporal': temporal_scores[idx],
				'combined': final_scores[idx]
			}
			results.append((int(idx), float(final_scores[idx]), score_breakdown))

		return results

	def split_recap_sentences(self, text: str) -> List[str]:
		"""sentence splitting with gemini"""
		cache_dir = f"{self.cache_path}/{re.sub(r'[^a-zA-Z]', '', text[:10])}_split_recap_sentences.json"
		if os.path.exists(cache_dir):
			logger_config.info("Using cache sentence splitting")
			with open(cache_dir, "r") as f:
				return json.load(f)

		logger_config.info("Starting sentence splitting")

		with open("sentence_split_system_prompt.md", 'r') as file:
			system_prompt = file.read()

		geminiWrapper = GeminiWrapper(system_instruction=system_prompt)
		model_responses = geminiWrapper.send_message(text, schema=genai.types.Schema(
			type = genai.types.Type.OBJECT,
			required = ["sentences"],
			properties = {
				"sentences": genai.types.Schema(
					type = genai.types.Type.ARRAY,
					items = genai.types.Schema(
						type = genai.types.Type.STRING,
					),
				),
			},
		))
		sentences = json.loads(model_responses[0])["sentences"]

		logger_config.info(f"Generated {len(sentences)} sentences")
		with open(cache_dir, 'w') as f:
			json.dump(sentences, f, indent=4)
		return sentences

	def process(self, input_json_path: str):
		"""Fullyprocessing pipeline"""
		logger_config.info("🚀 STARTING VIDEO-TEXT ALIGNMENT")
		overall_start = time.time()

		# Step 1: Setup
		with open(input_json_path, 'r') as file:
			input_json = json.load(file)

		video_path = input_json["video_path"]
		recap_text = input_json["text"]
		timestamp_data = input_json.get("timestamp_data", [])

		self.set_cache_dir(video_path)

		# Step 2: Load SentenceTransformer
		self.load_sentence_transformer()

		# Step 3: Load subtitles if provided
		self.load_subtitles(timestamp_data)

		# Step 4:scene extraction
		frame_paths, frame_numbers, timestamps = self.extract_scenes(video_path)

		self.load_clip_on_demand()
		# Step 5: Pre-compute CLIP embeddings
		frame_clip_embeddings = self.compute_frame_clip_embeddings(frame_paths)

		# Step 6:captioning
		captions = self.caption_generation(frame_paths, timestamps)

		# Step 7: Process text
		sentences = self.split_recap_sentences(recap_text)

		# Step 8:matching
		[f.unlink() for f in Path(TEMP_DIR).glob("sentence_*") if f.is_file()]
		results = []
		previous_timestamp = None

		for i, sentence in enumerate(sentences):
			matches = self.find_best_match(
				captions, timestamps, sentence, frame_clip_embeddings, 
				previous_timestamp, top_k=5
			)

			best_idx, best_score, score_breakdown = matches[0]
			timestamp = timestamps[best_idx]
			frame_num = frame_numbers[best_idx]

			# Save frame
			frame_second = frame_paths[best_idx].split("frame_second")[1]
			output_path = os.path.join(TEMP_DIR, f"sentence_{i+1:02d}_frame_{frame_num}_frame_second{frame_second}frame_second.jpg")
			import shutil
			shutil.copy2(frame_paths[best_idx], output_path)

			relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=2.0)

			results.append({
				'sentence_id': i + 1,
				'sentence': sentence,
				'frame_number': frame_num,
				'timestamp': timestamp,
				'caption': captions[best_idx],
				'score': best_score,
				'score_breakdown': score_breakdown,
				'output_path': output_path,
				'relevant_subtitles': relevant_subs
			})

			previous_timestamp = timestamp

			logger_config.info(f"Processed sentence {i+1}/{len(sentences)} - Score: {best_score:.4f}", overwrite=True)

		# Save results
		self.save_enhanced_results(results, OUTPUT_JSON)

		total_time = time.time() - overall_start
		avg_score = sum(r['score'] for r in results) / len(results) if results else 0

		logger_config.info(f"🎉 PROCESSING COMPLETE!")
		logger_config.info(f"⚡ Total time: {total_time:.2f}s | Avg score: {avg_score:.4f}")

		return results

	def reset(self):
		"""Reset with better memory management"""
		logger_config.info("Starting reset")

		# Clear caches
		self._sentence_cache.clear()
		self._clip_text_cache.clear()

		# Clear CUDA cache
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()

		# Force garbage collection
		gc.collect()

		# Reset temp directory
		if os.path.exists(TEMP_DIR):
			import shutil
			shutil.rmtree(TEMP_DIR)
		os.makedirs(TEMP_DIR, exist_ok=True)

		logger_config.info("Reset completed")

	def save_enhanced_results(self, results: List[dict], output_path: str):
		"""Results saving"""
		logger_config.info(f"Saving results to: {output_path}")

		json_output = []
		for result in results:
			frame_second = result['output_path'].split("frame_second")[1]
			json_entry = {
				"frame_path": os.path.abspath(result['output_path']),
				"frame_second": frame_second,
				"sentence": result['sentence'],
				"caption": result['caption'],
				"timestamp": result['timestamp'],
				"frame_number": result['frame_number'],
				"score_breakdown": result.get('score_breakdown', {}),
				"relevant_subtitles": result.get('relevant_subtitles', [])
			}
			json_output.append(json_entry)

		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(json_output, f, indent=2, ensure_ascii=False)

		logger_config.info(f"Results saved: {len(json_output)} entries")


# Usage example withversion
if __name__ == "__main__":
	# Initializematcher
	matcher = TextFrameAligner(max_workers=8)

	# Process with optimizations
	results = matcher.process("input.json")