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
import multiprocessing as mp
from functools import lru_cache
import gc
from gemiwrap import GeminiWrapper
from google import genai
import hashlib
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import shutil
from gemiwrap import GeminiWrapper
from extract_scenes import extract_scenes as extract_scenes_method
import traceback
from aistudio_ui_handler import run_gemini_generation

TEMP_DIR = os.path.abspath("temp_dir")
CACHE_DIR = f"{TEMP_DIR}/cache_dir"
OUTPUT_JSON = f'{TEMP_DIR}/output.json'

class TextFrameAligner:
	def __init__(self, 
				 blip_model_name="Salesforce/blip-image-captioning-large", 
				 sentence_model_name='all-mpnet-base-v2', 
				#  vision_model_name="google/siglip-base-patch16-224",  # Changed to SigLIP
				 vision_model_name="google/siglip2-so400m-patch16-384",  # Changed to SigLIP
				 max_workers=None):
		
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.max_workers = max_workers or max(8, mp.cpu_count()-4)
		logger_config.info("TextFrameAligner initialization started")
		logger_config.info(f"Compute device: {self.device}, Max workers: {self.max_workers}")
		
		# Environment setup with optimizations
		os.environ["HF_HUB_TIMEOUT"] = "120"
		if torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			torch.set_float32_matmul_precision('high')

		self.blip_model_name = blip_model_name
		self.sentence_model_name = sentence_model_name
		self.vision_model_name = vision_model_name
		self.cache_path = None

		# Model containers
		self.processor = None
		self.blip_model = None
		self.vision_processor = None
		self.vision_model = None
		self.embedder = None

		# Cached data
		self.subtitles = []
		self.subtitle_embeddings = None
		self._sentence_cache = {}
		self._vision_text_cache = {}
		self._tfidf_vectorizer = None
		
		# Fixed weights - rebalanced for better performance
		self.weights = {
			'vision': 0.45,	  # SigLIP/CLIP similarity
			'semantic': 0.40,	# Sentence transformer similarity
			'tfidf': 0.10,	   # TF-IDF similarity
			'subtitle': 0.20,	# Subtitle context similarity
			'temporal': 0.30	 # Temporal coherence
		}
		
		logger_config.info("TextFrameAligner initialization completed")

	def set_cache_dir(self, identifier: str) -> str:
		"""Generates a unique cache directory path for a given identifier."""
		content_hash = hashlib.md5(identifier.encode()).hexdigest()
		cache_path = os.path.join(CACHE_DIR, content_hash)
		if not os.path.exists(cache_path):
			self.reset()
		os.makedirs(cache_path, exist_ok=True)
		self.cache_path = cache_path

	def load_vision_model(self):
		"""Load SigLIP or CLIP model for vision-text matching"""
		if self.vision_model is not None:
			return

		logger_config.info(f"Loading vision model: {self.vision_model_name}")
		
		if "siglip" in self.vision_model_name.lower():
			from transformers import SiglipProcessor, SiglipModel
			self.vision_model = SiglipModel.from_pretrained(self.vision_model_name).to(self.device)
			self.vision_processor = SiglipProcessor.from_pretrained(self.vision_model_name)
		else:
			# Fallback to CLIP
			from transformers import CLIPProcessor, CLIPModel
			self.vision_model = CLIPModel.from_pretrained(self.vision_model_name).to(self.device)
			self.vision_processor = CLIPProcessor.from_pretrained(self.vision_model_name)

	@torch.inference_mode()
	def load_sentence_transformer(self):
		"""Load SentenceTransformer"""
		if self.embedder is not None:
			return
		logger_config.info("Loading SentenceTransformer")
		self.embedder = SentenceTransformer(self.sentence_model_name)
		if self.device == "cuda":
			self.embedder = self.embedder.to(self.device)
		logger_config.info("SentenceTransformer loaded successfully")

	def unload_sentence_transformer(self):
		"""Unload BLIP model to free memory"""
		if self.embedder:
			logger_config.info("Unloading SentenceTransformer model")
			del self.embedder
			self.embedder = None
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def load_blip_on_demand(self):
		"""Load BLIP model for image captioning"""
		if self.blip_model is not None:
			return

		logger_config.info(f"Loading BLIP model: {self.blip_model_name}")
		from transformers import BlipProcessor, BlipForConditionalGeneration
		self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
		self.blip_model = BlipForConditionalGeneration.from_pretrained(
			self.blip_model_name,
			torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
		).to(self.device)
		
		if self.device == "cuda":
			self.blip_model = torch.compile(self.blip_model, mode="reduce-overhead")

	def unload_blip(self):
		"""Unload BLIP model to free memory"""
		if self.blip_model:
			logger_config.info("Unloading BLIP model")
			del self.blip_model, self.processor
			self.blip_model = None
			self.processor = None
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	@torch.inference_mode()
	def compute_frame_vision_embeddings(self, frame_paths: List[str]) -> torch.Tensor:
		"""Compute vision embeddings (SigLIP/CLIP) with ETA and partial saves."""
		cache_dir = f"{self.cache_path}/compute_frame_vision_embeddings.pt"
		partial_dir = os.path.join(self.cache_path, "partial_embeddings")
		os.makedirs(partial_dir, exist_ok=True)

		# If full cache exists, return it
		if os.path.exists(cache_dir):
			logger_config.info("✅ Using cached vision embeddings")
			return torch.load(cache_dir)

		# Check for partial embeddings
		partial_embeddings = []
		already_processed = set()
		for i in range(len(frame_paths)):
			partial_path = os.path.join(partial_dir, f"embedding_{i:04d}.pt")
			if os.path.exists(partial_path):
				embedding = torch.load(partial_path)
				partial_embeddings.append(embedding)
				already_processed.add(i)

		logger_config.info(f"Resuming from partial cache: {len(already_processed)}/{len(frame_paths)} frames processed")

		all_embeddings = partial_embeddings
		times = []

		for i in range(len(frame_paths)):
			if i in already_processed:
				continue  # Skip already cached

			start_time = time.time()
			with Image.open(frame_paths[i]) as img:
				inputs = self.vision_processor(
					text=None, images=[img], return_tensors="pt", padding=True
				).to(self.device)

				with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32):
					image_features = self.vision_model.get_image_features(pixel_values=inputs.pixel_values)
					image_features = image_features.float()

				# Move to CPU and save partial embedding
				embedding_cpu = image_features.cpu()
				all_embeddings.append(embedding_cpu)

				partial_path = os.path.join(partial_dir, f"embedding_{i:04d}.pt")
				torch.save(embedding_cpu, partial_path)

			# Estimate duration
			elapsed = time.time() - start_time
			times.append(elapsed)
			avg_time = np.mean(times)
			remaining = avg_time * (len(frame_paths) - i - 1)
			eta = time.strftime("%H:%M:%S", time.gmtime(remaining))

			logger_config.info(
				f"Processed {i+1}/{len(frame_paths)} | ETA: {eta} | Frame: {frame_paths[i]}",
				overwrite=True
			)

			# Clean up GPU memory
			del inputs, image_features
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		# Combine all embeddings
		frame_vision_embeddings = torch.cat(all_embeddings, dim=0)

		# Save final combined embedding
		torch.save(frame_vision_embeddings, cache_dir)
		logger_config.info(f"✅ Saved full embeddings to {cache_dir}")

		return frame_vision_embeddings

	@lru_cache(maxsize=128)
	def get_cached_sentence_embeddings(self, text: str) -> torch.Tensor:
		"""Cache sentence embeddings"""
		return self.embedder.encode([text], convert_to_tensor=True)

	@lru_cache(maxsize=128)
	def get_cached_vision_text_embeddings(self, text: str) -> torch.Tensor:
		"""Cache vision model text embeddings"""
		inputs = self.vision_processor(text=[text], images=None, return_tensors="pt", padding=True).to(self.device)
		with torch.inference_mode():
			text_features = self.vision_model.get_text_features(**inputs)
		return text_features.cpu()

	def caption_generation(self, extract_scenes_json) -> List[str]:
		cache_dir = f"{self.cache_path}/caption_generation.json"
		partial_dir = os.path.join(self.cache_path, "partial_captions")
		os.makedirs(partial_dir, exist_ok=True)

		# If full cache exists, return it
		if os.path.exists(cache_dir):
			logger_config.info(f"✅ Using cached captions for {len(extract_scenes_json)} frames")
			with open(cache_dir, "r") as f:
				return json.load(f)

		logger_config.info(f"🚀 Starting caption generation for {len(extract_scenes_json)} frames")

		# Check for partial captions
		captions = []
		already_processed = set()
		for i in range(len(extract_scenes_json)):
			partial_path = os.path.join(partial_dir, f"caption_{i:04d}.json")
			if os.path.exists(partial_path):
				with open(partial_path, "r") as f:
					caption = json.load(f)
					captions.append({
						"scene_caption":caption.lower(),
						"scene_dialogue":extract_scenes_json[i]["dialogue"]
					})
					already_processed.add(i)

		logger_config.info(f"Resuming from partial cache: {len(already_processed)}/{len(extract_scenes_json)} frames processed")

		times = []
		# prompt = "What are these?"
		prompt = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context."
		from moondream2 import Moondream2
		from llava_one_vision import LlavaOneVision
		with Moondream2() as vision_model:
			for i in range(len(extract_scenes_json)):
				if i in already_processed:
					continue  # Skip already cached

				start_time = time.time()
				with Image.open(extract_scenes_json[i]["frame_path"][0]) as img:
					result = vision_model.generate(img, prompt)
					captions.append({
						"scene_caption":result.lower(),
						"scene_dialogue":extract_scenes_json[i]["dialogue"]
					})

					# Save partial caption
					partial_path = os.path.join(partial_dir, f"caption_{i:04d}.json")
					with open(partial_path, "w") as f:
						json.dump(result, f, indent=4)

				# Estimate duration
				elapsed = time.time() - start_time
				times.append(elapsed)
				avg_time = sum(times) / len(times)
				remaining = avg_time * (len(extract_scenes_json) - i)
				eta = time.strftime("%H:%M:%S", time.gmtime(remaining))

				logger_config.info(
					f"Caption Processed {i+1}/{len(extract_scenes_json)} | ETA: {eta}",
					overwrite=True
				)

		# Save final combined captions
		with open(cache_dir, "w") as f:
			json.dump(captions, f, indent=4)

		logger_config.info(f"✅ Saved all captions to {cache_dir}")
		return captions

	def compute_tfidf_similarity(self, captions: List[str], query: str) -> np.ndarray:
		"""Compute TF-IDF similarity with proper preprocessing"""
		try:
			# Preprocess text
			def preprocess_text(text):
				text = text.lower()
				text = text.translate(str.maketrans('', '', string.punctuation))
				return text
			
			processed_captions = [preprocess_text(cap) for cap in captions]
			processed_query = preprocess_text(query)
			
			# Create TF-IDF vectorizer
			if self._tfidf_vectorizer is None:
				self._tfidf_vectorizer = TfidfVectorizer(
					stop_words='english',
					ngram_range=(1, 2),
					max_features=5000,
					min_df=1,
					max_df=0.8,
					dtype=np.float32
				)
			
			# Fit on captions + query
			all_texts = processed_captions + [processed_query]
			tfidf_matrix = self._tfidf_vectorizer.fit_transform(all_texts)
			
			# Compute cosine similarity
			query_vector = tfidf_matrix[-1]
			caption_vectors = tfidf_matrix[:-1]
			
			similarities = cosine_similarity(query_vector, caption_vectors)[0]
			
			# Normalize to 0-1 range
			if similarities.max() > similarities.min():
				similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
			
			return similarities
			
		except Exception as e:
			logger_config.warning(f"TF-IDF computation failed: {e}")
			return np.zeros(len(captions))

	def compute_subtitle_similarity(self, timestamps: List[float], query: str) -> np.ndarray:
		"""Compute subtitle similarity scores"""
		if not self.subtitles or self.subtitle_embeddings is None:
			return np.zeros(len(timestamps))
		
		query_embedding = self.get_cached_sentence_embeddings(query)
		subtitle_scores = np.zeros(len(timestamps))
		
		for i, timestamp in enumerate(timestamps):
			relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=3.0)
			if relevant_subs:
				try:
					# Get subtitle indices
					sub_indices = []
					for sub in relevant_subs:
						try:
							idx = self.subtitles.index(sub)
							sub_indices.append(idx)
						except ValueError:
							continue
					
					if sub_indices:
						relevant_embeddings = self.subtitle_embeddings[sub_indices]
						similarities = util.cos_sim(query_embedding, relevant_embeddings)[0]
						subtitle_scores[i] = float(torch.max(similarities))
						
				except Exception as e:
					logger_config.warning(f"Subtitle similarity computation failed: {e}")
					continue
		
		return subtitle_scores

	def compute_temporal_coherence(self, timestamps: List[float], previous_timestamp: Optional[float] = None) -> np.ndarray:
		"""Compute temporal coherence scores"""
		temporal_scores = np.zeros(len(timestamps))
		
		if previous_timestamp is None:
			return temporal_scores
		
		timestamps_array = np.array(timestamps)
		time_diffs = timestamps_array - previous_timestamp
		
		# Forward progression bonus (prefer moving forward in time)
		forward_mask = time_diffs > 0
		temporal_scores[forward_mask] = np.maximum(
			0, 1.0 - (time_diffs[forward_mask] / 30.0)
		)
		
		# Backward penalty (discourage going backward)
		backward_mask = time_diffs <= 0
		temporal_scores[backward_mask] = -0.3 * np.minimum(
			1.0, np.abs(time_diffs[backward_mask]) / 10.0
		)
		
		return temporal_scores

	def similarity_computation(self, captions: List[str], timestamps: List[float], query: str, 
							 frame_vision_embeddings: torch.Tensor, 
							 previous_timestamp: Optional[float] = None) -> Dict[str, np.ndarray]:
		"""Compute all similarity metrics"""
		
		# 1. Semantic similarity (Sentence Transformer)
		query_semantic_emb = self.get_cached_sentence_embeddings(query)
		caption_embeddings = self.embedder.encode(captions, convert_to_tensor=True, batch_size=32)
		semantic_similarities = util.cos_sim(query_semantic_emb, caption_embeddings)[0].cpu().numpy()
		
		# 2. Vision similarity (SigLIP/CLIP)
		vision_similarities = 0.0
		if frame_vision_embeddings:
			query_vision_emb = self.get_cached_vision_text_embeddings(query)
			vision_similarities = util.cos_sim(query_vision_emb, frame_vision_embeddings)[0].cpu().numpy()
		
		# 3. TF-IDF similarity
		tfidf_similarities = self.compute_tfidf_similarity(captions, query)
		
		# 4. Subtitle similarity
		subtitle_similarities = self.compute_subtitle_similarity(timestamps, query)
		
		# 5. Temporal coherence
		temporal_similarities = self.compute_temporal_coherence(timestamps, previous_timestamp)
		
		return {
			'semantic': semantic_similarities,
			'vision': vision_similarities,
			'tfidf': tfidf_similarities,
			'subtitle': subtitle_similarities,
			'temporal': temporal_similarities
		}

	def load_subtitles(self, timestamp_data):
		"""Load subtitles with pre-computed embeddings"""
		logger_config.info(f"Loading timestamp data")

		if timestamp_data:
			self.subtitles = timestamp_data
			subtitle_texts = [sub['text'] for sub in self.subtitles]
			self.subtitle_embeddings = self.embedder.encode(
				subtitle_texts,
				convert_to_tensor=True,
				batch_size=64
			)
			logger_config.info(f"Pre-computed embeddings for {len(subtitle_texts)} subtitles")

	def get_subtitles_for_timerange(self, start_time: float, end_time: float, buffer: float = 2.0) -> List[Dict]:
		"""Get subtitles within time range"""
		if not self.subtitles:
			return []
		
		relevant_subs = []
		for sub in self.subtitles:
			if sub['end'] >= start_time - buffer and sub['start'] <= end_time + buffer:
				relevant_subs.append(sub)
		
		return relevant_subs

	def find_best_match(self, captions: List[str], timestamps: List[float], query: str, 
					   frame_vision_embeddings: torch.Tensor, 
					   previous_timestamp: Optional[float] = None, 
					   top_k: int = 5) -> List[Tuple[int, float, Dict]]:
		"""Find best matching frames with all similarity metrics"""

		# Compute all similarities
		similarities = self.similarity_computation(
			captions, timestamps, query, frame_vision_embeddings, previous_timestamp
		)

		# Combine scores using weights
		final_scores = (
			self.weights['semantic'] * similarities['semantic'] +
			self.weights['vision'] * similarities['vision'] +
			self.weights['tfidf'] * similarities['tfidf'] +
			self.weights['subtitle'] * similarities['subtitle'] +
			self.weights['temporal'] * similarities['temporal']
		)

		# Get top-k indices
		top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
		top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

		# Build results
		results = []
		for idx in top_indices:
			score_breakdown = {
				'semantic': float(similarities['semantic'][idx]),
				# 'vision': float(similarities['vision'][idx]),
				'tfidf': float(similarities['tfidf'][idx]),
				'subtitle': float(similarities['subtitle'][idx]),
				'temporal': float(similarities['temporal'][idx]),
				'combined': float(final_scores[idx])
			}
			results.append((int(idx), float(final_scores[idx]), score_breakdown))

		return results

	def split_recap_sentences(self, text: str) -> List[str]:
		"""Split text into sentences using Gemini"""
		cache_dir = f"{self.cache_path}/{re.sub(r'[^a-zA-Z]', '', text[:10])}_split_recap_sentences.json"
		if os.path.exists(cache_dir):
			logger_config.info("Using cached sentence splitting")
			with open(cache_dir, "r") as f:
				return json.load(f)

		logger_config.info("Starting sentence splitting")

		with open("sentence_split_system_prompt.md", 'r') as file:
			system_prompt = file.read()

		geminiWrapper = GeminiWrapper(system_instruction=system_prompt)
		model_responses = geminiWrapper.send_message(text, schema=genai.types.Schema(
			type=genai.types.Type.OBJECT,
			required=["sentences"],
			properties={
				"sentences": genai.types.Schema(
					type=genai.types.Type.ARRAY,
					items=genai.types.Schema(
						type=genai.types.Type.STRING,
					),
				),
			},
		))
		sentences = json.loads(model_responses[0])["sentences"]
		joined_processed = " ".join(sentences).strip()
		original = text.strip()

		import difflib
		similarity = difflib.SequenceMatcher(None, joined_processed, original).ratio()
		if similarity < 0.9:
			raise ValueError(f"process_narration_text is not correct:: similarity-{similarity}")

		logger_config.info(f"Generated {len(sentences)} sentences")
		with open(cache_dir, 'w') as f:
			json.dump(sentences, f, indent=4)
		return sentences

	def match_scenes_online(self, captions, sentences, extract_scenes_json):
		"""Optimized scene extraction"""
		match_scene = None
		cache_dir = f"{self.cache_path}/{re.sub(r'[^a-zA-Z]', '', sentences[0][:10])}_match_scenes_online.json"
		if os.path.exists(cache_dir):
			logger_config.info(f"Using cached match_scenes_online")
			with open(cache_dir, "r") as f:
				match_scene = json.load(f)

		if not match_scene:
			text = f"""Scene Captions:: {captions}
	Recap Sentences:: {sentences}"""
			with open("scene_matching_system_prompt.md", 'r') as file:
				system_prompt = file.read()

			# geminiWrapper = GeminiWrapper(system_instruction=system_prompt, model_name="gemini-2.0-flash")
			# model_responses = geminiWrapper.send_message(text, schema=genai.types.Schema(
			# 	type = genai.types.Type.OBJECT,
			# 	required = ["data"],
			# 	properties = {
			# 		"data": genai.types.Schema(
			# 			type = genai.types.Type.ARRAY,
			# 			items = genai.types.Schema(
			# 				type = genai.types.Type.OBJECT,
			# 				required = ["scene_caption", "recap_sentence"],
			# 				properties = {
			# 					"scene_caption": genai.types.Schema(
			# 						type = genai.types.Type.STRING,
			# 					),
			# 					"recap_sentence": genai.types.Schema(
			# 						type = genai.types.Type.STRING,
			# 					),
			# 				},
			# 			),
			# 		),
			# 	},
			# ))
			# match_scene = json.loads(model_responses[0])["data"]
			times = 5
			match_scene = None
			while times > 0 and match_scene is None:
				match_scene = run_gemini_generation(system_prompt, text)
				times -= 1
			if not match_scene:
				raise ValueError("failed to get match_scene.")
			# Cache results
			with open(cache_dir, "w") as f:
				json.dump(match_scene, f, indent=4)

		self.load_sentence_transformer()
		only_captions = [obj["scene_caption"] for obj in captions]
		captions_embeddings = self.embedder.encode(only_captions, convert_to_tensor=True)

		resulted_sentence = [sent["recap_sentence"] for sent in match_scene]
		resulted_sentences_embeddings = self.embedder.encode(resulted_sentence, convert_to_tensor=True)

		result = []
		max_sentences_len = max([len(sent) for sent in sentences]) + 20
		for i, curr_sent in enumerate(sentences):
			query_embedding = self.embedder.encode(curr_sent, convert_to_tensor=True)
			similarities = util.cos_sim(query_embedding, resulted_sentences_embeddings)
			resulted_idx = similarities.argmax()

			scene_caption = match_scene[resulted_idx]["scene_caption"]
			recap_sentence =  match_scene[resulted_idx]["recap_sentence"]
			if len(scene_caption) < len(recap_sentence):
				scene_caption =  match_scene[resulted_idx]["recap_sentence"]
				recap_sentence =  match_scene[resulted_idx]["scene_caption"]

			if len(recap_sentence) > max_sentences_len:
				os.remove(cache_dir)
				raise ValueError(f"Invalid sentence:: {recap_sentence}")

			query_embedding = self.embedder.encode(scene_caption, convert_to_tensor=True)
			similarities = util.cos_sim(query_embedding, captions_embeddings)
			frame_idx = similarities.argmax()

			result.append({
				"recap_sentence": curr_sent,
				"frame_second": extract_scenes_json[frame_idx]["scene_start"],
				"frame_path": extract_scenes_json[frame_idx]["frame_path"][0],
				"scene_caption": captions[frame_idx],
			})
			# Save frame
			output_path = os.path.join(TEMP_DIR, f"sentence_{i:02d}_frame_{frame_idx}.jpg")
			shutil.copy2(extract_scenes_json[frame_idx]["frame_path"][0], output_path)

			# Log progress
			logger_config.info(f"Aligned {i+1}/{len(sentences)} sentences")

		self.unload_sentence_transformer()
		return result

	def match_scenes_offline(self, captions, sentences, timestamps, frame_paths, timestamp_data):
		# Step 4: Load models
		self.load_sentence_transformer()

		# Step 5: Load subtitles
		self.load_subtitles(timestamp_data)

		# Step 6: Compute vision embeddings
		self.load_vision_model()
		frame_vision_embeddings = self.compute_frame_vision_embeddings(frame_paths)

		result = []
		for i, sentence in enumerate(sentences):
			sentence = sentence.lower()
			logger_config.info(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
			
			# Find best matches for this sentence
			matches = self.find_best_match(
				captions, timestamps, sentence, frame_vision_embeddings, 
				previous_timestamp, top_k=5
			)
			
			best_match = matches[0]
			best_idx, best_score, score_breakdown = best_match
			
			# Update previous timestamp for temporal coherence
			previous_timestamp = timestamps[best_idx]

			result.append({
				"recap_sentence": sentence,
				"frame_second": timestamps[best_idx],
				"scene_caption": captions[best_idx],
			})
			# Save frame
			output_path = os.path.join(TEMP_DIR, f"sentence_{i:02d}_frame_{best_idx}.jpg")
			shutil.copy2(frame_paths[best_idx], output_path)

			# Log progress
			logger_config.info(f"Aligned {i+1}/{len(sentences)} sentences")

		return result

	def process(self, input_json_path: str):
		"""Full processing pipeline"""
		logger_config.info("🚀 STARTING VIDEO-TEXT ALIGNMENT")
		overall_start = time.time()

		# Step 1: Setup
		with open(input_json_path, 'r') as file:
			input_json = json.load(file)

		video_path = input_json["video_path"]
		recap_text = input_json["text"]
		frame_timestamp = input_json.get("frame_timestamp", [])
		frame_paths = input_json.get("frame_paths", [])

		self.set_cache_dir(video_path)

		if not frame_paths:
			# Step 2: Extract scenes
			extract_scenes_json = extract_scenes_method(video_path, frame_timestamp, self.cache_path, os.path.join(TEMP_DIR, "frames"))
		else:
			extract_scenes_json = []
			for path in frame_paths:
				extract_scenes_json.append(
					{
						"frame_path": [path],
						"scene_start": 0
					}
				)

		# Step 3: Generate captions
		captions = self.caption_generation(extract_scenes_json)

		# Step 7: Process text
		sentences = self.split_recap_sentences(recap_text)

		# Step 8: Align sentences to frames
		logger_config.info("🎯 STARTING SENTENCE-FRAME ALIGNMENT")
		[f.unlink() for f in Path(TEMP_DIR).glob("sentence_*") if f.is_file()]

		sentences = [s.lower() for s in sentences]

		try_times = 2
		result = None

		while (result is None or len(result) == 0) and try_times > 0:
			try:
				result = self.match_scenes_online(captions, sentences, extract_scenes_json)
				# result = self.match_scenes_offline(captions, sentences, timestamps, frame_paths, frame_numbers, timestamp_data)
			except Exception as e:
				logger_config.error(f'{str(e)} {traceback.format_exc()}')
				try_times -=1

		if (result is None or len(result) == 0):
			raise ValueError("Result is empty")
		
		# Save output
		with open(OUTPUT_JSON, 'w') as f:
			json.dump(result, f, indent=4)
		
		logger_config.info(f"✅ ALIGNMENT COMPLETE! Results saved to {OUTPUT_JSON}")
		logger_config.info(f"⏱️ Total processing time: {time.time() - overall_start:.2f} seconds")

		return result

	def reset(self):
		"""Reset cached data and free memory"""
		logger_config.info("Resetting TextFrameAligner")
		shutil.rmtree(TEMP_DIR, ignore_errors=True)
		os.makedirs(TEMP_DIR, exist_ok=True)

		# Clear cached data
		self.subtitles = []
		self.subtitle_embeddings = None
		self._sentence_cache.clear()
		self._vision_text_cache.clear()
		self._tfidf_vectorizer = None
		
		# Unload models
		self.unload_blip()
		
		if self.vision_model is not None:
			del self.vision_model, self.vision_processor
			self.vision_model = None
			self.vision_processor = None
		
		if self.embedder is not None:
			del self.embedder
			self.embedder = None
		
		# Clear GPU memory
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		# Force garbage collection
		gc.collect()
		
		logger_config.info("Reset complete")

	def cleanup(self):
		"""Clean up model and related resources."""
		try:
			print("Cleaning up resources...")

			# Dereference all components
			self.processor = None
			self.blip_model = None
			self.vision_processor = None
			self.vision_model = None
			self.embedder = None

			if hasattr(self, 'processor') and self.processor is not None:
				self.processor = None
				print("processor reference cleared.")
			if hasattr(self, 'vision_processor') and self.vision_processor is not None:
				self.vision_processor = None
				print("vision_processor reference cleared.")
			if hasattr(self, 'vision_model') and self.vision_model is not None:
				self.vision_model = None
				print("vision_model reference cleared.")
			if hasattr(self, 'embedder') and self.embedder is not None:
				self.embedder = None
				print("embedder reference cleared.")

			# Run garbage collection
			gc.collect()

			# Free CUDA memory if applicable
			try:
				import torch
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
					print("CUDA memory cleaned.")
			except ImportError:
				print("Torch not available; skipped GPU cleanup.")

			print("Cleanup completed successfully.")

		except Exception as e:
			print(f"Error during cleanup: {e}")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.cleanup()

	def __del__(self):
		self.cleanup()

# Usage example and main execution
if __name__ == "__main__":
	# Example usage
	aligner = TextFrameAligner()
	
	try:
		# Process video-text alignment
		results = aligner.process("input.json")
		
	except Exception as e:
		logger_config.error(f"Processing failed: {e}")
		raise