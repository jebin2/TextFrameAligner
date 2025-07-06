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

TEMP_DIR = "temp_dir"
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
			'semantic': 0.35,
			'tfidf': 0.15,  # Reduced as it's computationally expensive
			'subtitle': 0.25,
			'temporal': 0.05
		}
		
		logger_config.info("TextFrameAligner initialization completed")

	@torch.inference_mode()
	def load_models_batch(self):
		"""Load all models at once to avoid repeated loading/unloading"""
		logger_config.info("Loading all models in batch for optimal memory usage")

		# Load CLIP
		from transformers import CLIPProcessor, CLIPModel
		self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
		self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

		# Load SentenceTransformer
		self.embedder = SentenceTransformer(self.sentence_model_name)
		if self.device == "cuda":
			self.embedder = self.embedder.to(self.device)

		logger_config.info("All models loaded successfully")

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

	def is_mostly_black(self, img, black_pixel_threshold=0.9, black_rgb_threshold=10):
		"""
		Returns True if >= 90% of pixels are near black.
		`black_rgb_threshold` defines how dark a pixel must be to count as black.
		"""
		img = img.convert("RGB")
		pixels = list(img.getdata())
		total_pixels = len(pixels)
		black_pixels = sum(
			1 for r, g, b in pixels if r <= black_rgb_threshold and g <= black_rgb_threshold and b <= black_rgb_threshold
		)
		return (black_pixels / total_pixels) >= black_pixel_threshold

	def extract_scenes(self, video_path: str, threshold: float = 30.0) -> Tuple[List[str], List[int], List[float]]:
		"""scene extraction with parallel frame processing"""
		import cv2
		from scenedetect import detect, ContentDetector
		logger_config.info(f"Starting scene detection: {video_path}")

		# Scene detection
		scene_list = detect(video_path, ContentDetector(threshold=threshold), show_progress=True)
		logger_config.info(f"Found {len(scene_list)} scenes")

		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps <= 0:
			raise ValueError("Could not read FPS from video")

		frames_dir = os.path.join(TEMP_DIR, "frames")
		os.makedirs(frames_dir, exist_ok=True)

		# Prepare frame extraction tasks
		extraction_tasks = []
		for i, (start_time, end_time) in enumerate(scene_list):
			start_frame = start_time.get_frames()
			extraction_tasks.append((i, start_frame, start_frame / fps))

		cap.release()

		# Parallel frame extraction
		def extract_single_frame(task):
			i, frame_num, timestamp = task
			cap_local = cv2.VideoCapture(video_path)
			cap_local.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
			ret, frame = cap_local.read()
			cap_local.release()

			if not ret:
				return None

			with Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) as image:
				if self.is_mostly_black(image):
					return None
				filename = f"scene_{i:03d}_frame_{frame_num}_at_frame_second{timestamp:.2f}frame_second.jpg"
				frame_path = os.path.join(frames_dir, filename)

				# Optimize image saving
				image.save(frame_path, format='JPEG', quality=85, optimize=True)
				logger_config.info(f"Save:: {frame_path}", overwrite=True)

			return frame_path, frame_num, timestamp

		# Use ThreadPoolExecutor for I/O bound operations
		frame_paths, frame_numbers, timestamps = [], [], []
		logger_config.info("For overwrite")
		with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			results = list(executor.map(extract_single_frame, extraction_tasks))

		for result in results:
			if result:
				fp, fn, ts = result
				frame_paths.append(fp)
				frame_numbers.append(fn)
				timestamps.append(ts)

		logger_config.info(f"Extracted {len(frame_paths)} frames in parallel")
		return frame_paths, frame_numbers, timestamps

	@torch.inference_mode()
	def compute_frame_clip_embeddings(self, frame_paths: List[str]) -> torch.Tensor:
		"""Highly CLIP embedding computation with memory management"""
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

		return torch.cat(all_embeddings, dim=0)

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
		
		self.subtitles = timestamp_data
		
		# Pre-compute embeddings in batch
		if self.subtitles:
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
		return sentences

	def process(self, input_json_path: str):
		"""Fullyprocessing pipeline"""
		logger_config.info("🚀 STARTING VIDEO-TEXT ALIGNMENT")
		overall_start = time.time()

		# Step 1: Setup
		self.reset()

		with open(input_json_path, 'r') as file:
			input_json = json.load(file)

		video_path = input_json["video_path"]
		recap_text = input_json["text"]
		timestamp_data = input_json["timestamp_data"]

		# Step 2: Load all models at once
		self.load_models_batch()

		# Step 3: Load subtitles if provided
		if timestamp_data:
			self.load_subtitles(timestamp_data)

		# Step 4:scene extraction
		frame_paths, frame_numbers, timestamps = self.extract_scenes(video_path)

		# Step 5: Pre-compute CLIP embeddings
		frame_clip_embeddings = self.compute_frame_clip_embeddings(frame_paths)

		# Step 6:captioning
		captions = self.caption_generation(frame_paths, timestamps)

		# Step 7: Process text
		sentences = self.split_recap_sentences(recap_text)

		# Step 8:matching
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