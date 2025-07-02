from custom_logger import logger_config
import cv2
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import re
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import time
import json
import difflib

TEMP_DIR = "temp_dir"
OUTPUT_JSON = f'{TEMP_DIR}/output.json'

class TextFrameAligner:
	def __init__(self, blip_model_name="Salesforce/blip-image-captioning-large", sentence_model_name='all-mpnet-base-v2', clip_model_name="openai/clip-vit-large-patch14"):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logger_config.info("TextFrameAligner initialization started")
		logger_config.info(f"Compute device selected: {self.device}")
		
		# Environment setup
		# os.environ["TORCH_USE_CUDA_DSA"] = "1"
		# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		os.environ["HF_HUB_TIMEOUT"] = "120"
		logger_config.debug("CUDA environment variables configured")

		self.blip_model_name = blip_model_name
		self.sentence_model_name = sentence_model_name
		self.clip_model_name = clip_model_name

		self.processor = None
		self.blip_model = None
		self.clip_processor = None
		self.clip_model = None
		self.embedder = None
		self.nlp = None

		self.subtitles = []
		self.subtitle_embeddings = None
		
		# Improved matching weights
		self.weights = {
            'clip': 0.40,      # Direct text-to-image similarity
			'semantic': 0.35,
			'tfidf': 0.20,
			'entity': 0.15,
			'subtitle': 0.25,  # New subtitle matching
			'temporal': 0.05   # Temporal coherence bonus
		}
		logger_config.info("Matching weights configured: semantic=0.35, tfidf=0.20, entity=0.15, subtitle=0.25, temporal=0.05")
		
		logger_config.info("TextFrameAligner initialization completed successfully")

	def load_blip(self):
		if self.blip_model is not None:
			return  # Already loaded

		logger_config.info(f"Loading BLIP model: {self.blip_model_name}")
		from transformers import BlipProcessor, BlipForConditionalGeneration
		self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
		self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name).to(self.device)

	def unload_blip(self):
		if self.blip_model:
			logger_config.info("Unloading BLIP model to free memory")
			del self.blip_model
			del self.processor
			self.blip_model = None
			self.processor = None

	def load_clip(self):
		if self.clip_model is not None:
			return  # Already loaded

		logger_config.info(f"Loading CLIP model: {self.clip_model_name}")
		from transformers import CLIPProcessor, CLIPModel
		self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
		self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

	def unload_clip(self):
		if self.clip_model:
			logger_config.info("Unloading CLIP model to free memory")
			del self.clip_model
			del self.clip_processor
			self.clip_model = None
			self.clip_processor = None

	def load_sentence_model(self):
		if self.embedder is not None:
			return  # Already loaded

		logger_config.info(f"Loading SentenceTransformer model: {self.sentence_model_name}")
		self.embedder = SentenceTransformer(self.sentence_model_name)

	def unload_sentence_model(self):
		if self.embedder:
			logger_config.info("Unloading SentenceTransformer to free memory")
			del self.embedder
			self.embedder = None

	def load_spacy(self):
		"""Load spaCy model if not already loaded."""
		if self.nlp is not None:
			return

		import spacy
		try:
			logger_config.info("Loading spaCy NLP model: en_core_web_sm")
			self.nlp = spacy.load("en_core_web_sm")
		except OSError:
			logger_config.warning("spaCy model not found locally, downloading it...")
			from spacy.cli import download
			download("en_core_web_sm")
			self.nlp = spacy.load("en_core_web_sm")

	def unload_spacy(self):
		"""Unload spaCy NLP model to free memory."""
		if self.nlp:
			logger_config.info("Unloading spaCy NLP model to free memory")
			del self.nlp
			self.nlp = None

	def load_subtitles(self, subtitle_path: str):
		"""Load and process subtitle data for enhanced matching"""
		logger_config.info(f"Loading subtitle data from: {subtitle_path}")

		with open(subtitle_path, 'r', encoding='utf-8') as f:
			self.subtitles = json.load(f)

		logger_config.info(f"Loaded {len(self.subtitles)} subtitle entries from file")
		
		# Create subtitle text corpus for embedding
		subtitle_texts = [sub['text'] for sub in self.subtitles]
		
		if subtitle_texts:
			self.load_sentence_model()
			logger_config.info("Computing sentence embeddings for subtitle corpus")
			self.subtitle_embeddings = self.embedder.encode(subtitle_texts, convert_to_tensor=True)
			logger_config.info(f"Generated embeddings for {len(subtitle_texts)} subtitle entries")
			self.unload_sentence_model()
		
		# Log subtitle coverage
		if self.subtitles:
			start_time = min(sub['start'] for sub in self.subtitles)
			end_time = max(sub['end'] for sub in self.subtitles)
			duration = end_time - start_time
			logger_config.info(f"Subtitle temporal coverage: {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s)")

	def get_subtitles_for_timerange(self, start_time: float, end_time: float, buffer: float = 2.0) -> List[Dict]:
		"""Get subtitles that overlap with a given time range"""
		relevant_subs = []
		
		for sub in self.subtitles:
			# Check if subtitle overlaps with the time range (with buffer)
			if (sub['end'] >= start_time - buffer and sub['start'] <= end_time + buffer):
				relevant_subs.append(sub)
		
		logger_config.debug(f"Found {len(relevant_subs)} subtitles overlapping with timerange {start_time:.1f}-{end_time:.1f}s (buffer: {buffer}s)", overwrite=True)
		return relevant_subs

	def extract_scenes_with_scenedetect(self, video_path: str, threshold: float = 30.0) -> Tuple[List[str], List[int], List[float]]:
		"""Extract one frame per scene using modern PySceneDetect API."""
		from scenedetect import detect, ContentDetector, ThresholdDetector
		logger_config.info(f"Starting scene detection for video: {video_path}")
		logger_config.info(f"Scene detection threshold set to: {threshold}")

		# Step 1: Detect scenes
		logger_config.info("Analyzing video content for scene boundaries")
		scene_list = detect(video_path, ContentDetector(threshold=threshold))
		logger_config.info(f"Scene detection completed. Found {len(scene_list)} distinct scenes")

		# Step 2: Prepare for frame extraction
		logger_config.info("Opening video file for frame extraction")
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps <= 0:
			raise ValueError("Could not read FPS from video file")
		
		logger_config.info(f"Video FPS detected: {fps}")

		frames_dir = os.path.join(TEMP_DIR, "frames")
		os.makedirs(frames_dir, exist_ok=True)
		logger_config.info(f"Frame extraction directory prepared: {frames_dir}")

		frame_paths = []
		frame_numbers = []
		timestamps = []

		logger_config.info("Starting frame extraction from scene boundaries")
		for i, (start_time, end_time) in enumerate(scene_list):
			start_frame = start_time.get_frames()
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
			ret, frame = cap.read()
			if not ret:
				logger_config.warning(f"Failed to extract frame at position {start_frame} for scene {i}")
				continue

			image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			filename = f"scene_{i:03d}_frame_{start_frame}.jpg"
			frame_path = os.path.join(frames_dir, filename)
			image.save(frame_path, quality=85)

			frame_paths.append(frame_path)
			frame_numbers.append(start_frame)
			timestamps.append(start_frame / fps)

			logger_config.info(f"Frame extraction progress: {i + 1}/{len(scene_list)} scenes processed", overwrite=True)

		cap.release()
		logger_config.info(f"Frame extraction completed. Extracted {len(frame_paths)} frames from {len(scene_list)} scenes")
		return frame_paths, frame_numbers, timestamps

	def compute_frame_clip_embeddings(self, frame_paths: List[str]) -> torch.Tensor:
		"""Efficiently compute CLIP image embeddings for all frames in batches."""
		self.load_clip()
		logger_config.info("Computing CLIP image embeddings for all frames.")
		all_embeddings = []
		batch_size = 16  # Adjust based on your VRAM

		for i in range(0, len(frame_paths), batch_size):
			batch_paths = frame_paths[i:i + batch_size]
			images = [Image.open(p) for p in batch_paths]

			inputs = self.clip_processor(text=None, images=images, return_tensors="pt", padding=True).to(self.device)
			with torch.inference_mode():
				image_features = self.clip_model.get_image_features(inputs.pixel_values)

			all_embeddings.append(image_features.cpu())

			for img in images: # Close images after processing
				img.close()

			logger_config.debug(f"Processed batch {i//batch_size + 1}/{(len(frame_paths)-1)//batch_size + 1} for CLIP embeddings.", overwrite=True)
		self.unload_clip()
		return torch.cat(all_embeddings, dim=0)

	def enhanced_caption_generation(self, frame_paths: List[str], timestamps: List[float]) -> List[str]:
		"""Enhanced captioning that incorporates subtitle context"""
		self.load_blip()
		self.load_sentence_model()
		frame_count = len(frame_paths)
		logger_config.info(f"Starting caption generation for {frame_count} extracted frames")
		
		captions = []
		
		for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
			try:
				frame_start_time = time.time()
				image = Image.open(frame_path)
				
				# Get relevant subtitles for context
				relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=1.0)
				context_text = " ".join([sub['text'] for sub in relevant_subs])
				
				# Generate caption with context if available
				if context_text.strip():
					logger_config.debug(f"Frame {i+1}/{frame_count}: Using subtitle context for caption generation")
					# Use context-aware prompting
					inputs = self.processor(
						images=image, 
						text=f"A scene where {context_text[:100]}...",  # Truncate for token limits
						return_tensors="pt"
					).to(self.device)
				else:
					logger_config.debug(f"Frame {i+1}/{frame_count}: Generating caption without subtitle context")
					inputs = self.processor(images=image, return_tensors="pt").to(self.device)
				
				with torch.inference_mode():
					out = self.blip_model.generate(
						**inputs, 
						max_length=100,  # Increased for richer descriptions
						num_beams=8,
						length_penalty=0.8,
						repetition_penalty=1.2,
						# do_sample=True,
						# temperature=0.7
					)
				
				caption = self.processor.decode(out[0], skip_special_tokens=True)
				
				# Remove context text if it was prepended
				if context_text.strip():
					caption = caption.replace(f"A scene where {context_text[:100]}...", "").strip()
				
				captions.append(caption)
				image.close()
				frame_time = time.time() - frame_start_time
				
				if (i + 1) % 10 == 0:
					progress = (i + 1) / len(frame_paths) * 100
					logger_config.info(f"Caption generation progress: {i + 1}/{len(frame_paths)} frames ({progress:.1f}%)")

				logger_config.debug(f"Frame {i+1} at {timestamp:.2f}s: Caption generated in {frame_time:.2f}s")
				logger_config.debug(f"Generated caption: '{caption}'")
				if context_text:
					logger_config.debug(f"Used subtitle context: '{context_text[:50]}...'")
					
			except Exception as e:
				raise ValueError(f"Caption generation failed for frame {frame_path}: {str(e)}")
		
		logger_config.info(f"Caption generation completed for all {len(captions)} frames")
		self.unload_blip()
		self.unload_sentence_model()
		return captions

	def subtitle_similarity_score(self, query: str, timestamp: float, buffer: float = 3.0) -> float:
		"""Calculate similarity between query and subtitles around a timestamp"""
		if not self.subtitles or self.subtitle_embeddings is None:
			logger_config.debug(f"No subtitles available for similarity calculation at {timestamp:.1f}s")
			return 0.0
		
		# Get subtitles around the timestamp
		relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer)
		
		if not relevant_subs:
			logger_config.debug(f"No relevant subtitles found around timestamp {timestamp:.1f}s")
			return 0.0
		
		# Get indices of relevant subtitles
		relevant_indices = []
		for sub in relevant_subs:
			try:
				idx = self.subtitles.index(sub)
				relevant_indices.append(idx)
			except ValueError:
				continue
		
		if not relevant_indices:
			logger_config.debug(f"Could not map relevant subtitles to indices at {timestamp:.1f}s")
			return 0.0
		
		# Compute similarity with relevant subtitle embeddings
		query_embedding = self.embedder.encode([query], convert_to_tensor=True)
		relevant_embeddings = self.subtitle_embeddings[relevant_indices]
		
		similarities = util.cos_sim(query_embedding, relevant_embeddings)[0]
		max_similarity = float(torch.max(similarities))
		
		logger_config.debug(f"Subtitle similarity score at {timestamp:.1f}s: {max_similarity:.4f} (compared with {len(relevant_indices)} subtitle entries)")
		return max_similarity

	def temporal_coherence_score(self, current_timestamp: float, previous_timestamp: Optional[float]) -> float:
		"""Give bonus for temporal coherence (sequential story progression)"""
		if previous_timestamp is None:
			logger_config.debug("No previous timestamp available for temporal coherence calculation")
			return 0.0
		
		time_diff = abs(current_timestamp - previous_timestamp)
		
		# Prefer forward progression, penalize large jumps
		if current_timestamp > previous_timestamp:
			# Forward progression bonus, decaying with time difference
			score = max(0, 1.0 - (time_diff / 30.0))  # 30 second decay
			logger_config.debug(f"Forward temporal progression: {previous_timestamp:.1f}s -> {current_timestamp:.1f}s (diff: {time_diff:.1f}s, score: {score:.4f})", overwrite=True)
		else:
			# Backward jump penalty
			score = -0.5 * min(1.0, time_diff / 10.0)
			logger_config.debug(f"Backward temporal jump: {previous_timestamp:.1f}s -> {current_timestamp:.1f}s (diff: {time_diff:.1f}s, penalty: {score:.4f})", overwrite=True)
		
		return score

	def find_best_match(self, captions: List[str], timestamps: List[float], query: str, frame_clip_embeddings: torch.Tensor, previous_timestamp: Optional[float] = None, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
		"""Enhanced matching with subtitle integration and temporal coherence"""
		logger_config.debug(f"Starting enhanced matching for query: '{query[:100]}...'")

		# 0. CLIP Similarity (Text-to-Image)
		logger_config.debug("Computing CLIP similarity scores.")
		clip_inputs = self.clip_processor(text=[query], images=None, return_tensors="pt", padding=True).to(self.device)
		with torch.inference_mode():
			text_features = self.clip_model.get_text_features(**clip_inputs)
		clip_similarities = util.cos_sim(text_features.cpu(), frame_clip_embeddings)[0]

		# 1. Semantic similarity
		logger_config.debug("Computing semantic similarity scores using sentence embeddings")
		all_texts = captions + [query]
		embeddings = self.embedder.encode(all_texts, convert_to_tensor=True)
		caption_embs = embeddings[:-1]
		query_emb = embeddings[-1:]
		semantic_similarities = util.cos_sim(query_emb, caption_embs)[0]
		
		# 2. TF-IDF similarity
		logger_config.debug("Computing TF-IDF similarity scores")
		try:
			from sklearn.feature_extraction.text import TfidfVectorizer
			from sklearn.metrics.pairwise import cosine_similarity
			vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
			tfidf_matrix = vectorizer.fit_transform(captions + [query])
			tfidf_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
			logger_config.debug("TF-IDF similarity computation completed")
		except Exception as e:
			logger_config.warning(f"TF-IDF similarity computation failed: {str(e)}, using zero scores")
			tfidf_similarities = np.zeros(len(captions))
		
		# 3. Entity matching
		# logger_config.debug("Computing entity matching scores")
		# entity_scores = self.enhance_matching_with_entities(captions, query)
		
		# 4. Subtitle similarity scores
		logger_config.debug("Computing subtitle similarity scores for all timestamps")
		subtitle_scores = [self.subtitle_similarity_score(query, ts) for ts in timestamps]
		
		# 5. Temporal coherence scores
		logger_config.debug("Computing temporal coherence scores")
		temporal_scores = [self.temporal_coherence_score(ts, previous_timestamp) for ts in timestamps]
		
		# Combine all scores
		logger_config.debug("Combining all similarity scores with weighted average")
		final_scores = []
		for i in range(len(captions)):
			combined_score = (
				self.weights['clip'] * float(clip_similarities[i]) +
				self.weights['semantic'] * float(semantic_similarities[i]) +
				self.weights['tfidf'] * float(tfidf_similarities[i]) +
				# self.weights['entity'] * entity_scores[i] +
				self.weights['subtitle'] * subtitle_scores[i] +
				self.weights['temporal'] * temporal_scores[i]
			)
			
			score_breakdown = {
				'clip': float(clip_similarities[i]),
				'semantic': float(semantic_similarities[i]),
				'tfidf': float(tfidf_similarities[i]),
				# 'entity': entity_scores[i],
				'subtitle': subtitle_scores[i],
				'temporal': temporal_scores[i],
				'combined': combined_score
			}
			
			final_scores.append((i, combined_score, score_breakdown))
		
		# Sort and return top-k
		final_scores.sort(key=lambda x: x[1], reverse=True)
		logger_config.debug(f"Enhanced matching completed. Returning top {top_k} matches")
		
		# Log top matches
		for rank, (idx, score, breakdown) in enumerate(final_scores[:3]):
			logger_config.debug(f"Match rank {rank+1}: Frame {idx} at {timestamps[idx]:.1f}s (score: {score:.4f})")

		return final_scores[:top_k]

	# def enhance_matching_with_entities(self, captions: List[str], query: str) -> List[float]:
	# 	"""Enhanced entity matching with character names and anime-specific terms"""
		# logger_config.debug("Starting entity extraction from query text")
		# doc = self.nlp(query)
		
		# Extract entities with expanded categories
		# entities = []
		# for ent in doc.ents:
		# 	if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']:
		# 		entities.append(ent.text.lower())
		
		# logger_config.debug(f"Extracted {len(entities)} named entities: {entities}")
		
		# Extract important nouns and proper nouns
		# important_terms = []
		# for token in doc:
		# 	if (token.pos_ in ['NOUN', 'PROPN'] and 
		# 		not token.is_stop and 
		# 		len(token.text) > 2):
		# 		important_terms.append(token.text.lower())
		
		# logger_config.debug(f"Extracted {len(important_terms)} important terms: {important_terms[:10]}...")
		
		# Add anime/action specific terms
		# action_terms = ['attack', 'power', 'energy', 'beam', 'blast', 'fight', 'battle', 'strike', 'punch', 'kick', 'transformation', 'aura']
		# query_lower = query.lower()
		# action_terms_found = []
		# for term in action_terms:
		# 	if term in query_lower:
		# 		important_terms.append(term)
		# 		action_terms_found.append(term)
		
		# if action_terms_found:
		# 	logger_config.debug(f"Found action-specific terms: {action_terms_found}")
		
		# key_terms = list(set(entities + important_terms))
		# logger_config.debug(f"Total key terms for matching: {len(key_terms)}")
		
		# if not key_terms:
		# 	logger_config.debug("No key terms found, returning zero scores for all captions")
		# 	return [0.0] * len(captions)
		
		# Calculate fuzzy matching scores
		# logger_config.debug("Computing entity matching scores with fuzzy matching")
		# keyword_scores = []
		# for i, caption in enumerate(captions):
			# caption_lower = caption.lower()
			# exact_matches = sum(1 for term in key_terms if term in caption_lower)
			
			# Add fuzzy matching for character names
			# fuzzy_matches = 0
			# for term in key_terms:
			# 	if len(term) > 4:  # Only for longer terms
			# 		matches = difflib.get_close_matches(term, caption_lower.split(), n=1, cutoff=0.8)
			# 		if matches:
			# 			fuzzy_matches += 0.5  # Half score for fuzzy matches
			
			# total_score = exact_matches + fuzzy_matches
			# keyword_scores.append(total_score)
			
			# if total_score > 0:
			# 	logger_config.debug(f"Caption {i}: {exact_matches} exact + {fuzzy_matches:.1f} fuzzy matches = {total_score:.1f}", overwrite=True)
		
		# Normalize scores
		# max_score = max(keyword_scores) if keyword_scores else 1
		# normalized_scores = [score / max_score if max_score > 0 else 0 for score in keyword_scores]
		
		# logger_config.debug(f"Entity matching completed. Max raw score: {max_score}, normalized scores computed")
		# return normalized_scores

	def reset(self):
		"""Reset environment and clean up temporary files"""
		logger_config.info("Starting environment reset and cleanup")
		
		import torch
		if torch.cuda.is_available():
			logger_config.debug("Clearing CUDA cache and synchronizing")
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		
		if os.path.exists(TEMP_DIR):
			import shutil
			logger_config.info(f"Removing existing temporary directory: {TEMP_DIR}")
			shutil.rmtree(TEMP_DIR)
		
		os.makedirs(TEMP_DIR, exist_ok=True)
		logger_config.info(f"Created fresh temporary directory: {TEMP_DIR}")

	def clean_sentence(self, sentence: str) -> str:
		"""Enhanced sentence cleaning"""
		logger_config.debug(f"Cleaning sentence: '{sentence[:50]}...'")
		
		# Remove excessive punctuation and normalize
		sentence = re.sub(r'[—…!?]+', '.', sentence)
		sentence = re.sub(r'\s+', ' ', sentence)
		sentence = re.sub(r'\.{2,}', '.', sentence)  # Multiple dots to single
		cleaned = sentence.strip()
		
		logger_config.debug(f"Sentence cleaned: '{cleaned[:50]}...'")
		return cleaned

	def split_recap_sentences(self, text: str) -> List[str]:
		"""Enhanced sentence splitting with better handling"""
		logger_config.info("Starting recap text sentence splitting")
		logger_config.debug(f"Input text length: {len(text)} characters")
		
		cleaned_text = self.clean_sentence(text)
		logger_config.debug("Text cleaning completed")

		self.load_spacy()
		doc = self.nlp(cleaned_text)
		self.unload_spacy()
		logger_config.debug("spaCy sentence segmentation completed")
		
		sentences = []
		for sent in doc.sents:
			sent_text = sent.text.strip()
			# More flexible minimum length
			if len(sent_text) > 8 and not sent_text.endswith('...'):
				sentences.append(sent_text)
				logger_config.debug(f"Accepted sentence: '{sent_text[:50]}...'")
			else:
				logger_config.debug(f"Rejected sentence (too short or incomplete): '{sent_text}'")
		
		logger_config.info(f"Sentence splitting completed. Generated {len(sentences)} valid sentences")
		return sentences

	def save_enhanced_results(self, results: List[dict], output_path: str):
		"""Save enhanced results with additional metadata"""
		logger_config.info(f"Saving results to JSON file: {output_path}")
		
		json_output = []
		for i, result in enumerate(results):
			json_entry = {
				"frame_path": os.path.abspath(result['output_path']),
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
		
		logger_config.info(f"Results successfully saved. {len(json_output)} entries written to {output_path}")

	def process(self, video_path: str, recap_text: str, subtitle_path: str = None):
		"""Enhanced processing with subtitle integration"""
		logger_config.info("STARTING ENHANCED VIDEO-TEXT ALIGNMENT PROCESS")
		
		overall_start_time = time.time()
		logger_config.info(f"Process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
		logger_config.info(f"Input video: {video_path}")
		logger_config.info(f"Recap text length: {len(recap_text)} characters")
		
		# Step 1: Reset and environment setup
		logger_config.info("STEP 1: Environment Setup and Reset")
		self.reset()
		
		# Step 2: Load subtitles if provided
		if subtitle_path:
			logger_config.info("STEP 2: Loading Subtitle Data")
			logger_config.info(f"Subtitle file: {subtitle_path}")
			self.load_subtitles(subtitle_path)
		else:
			logger_config.info("STEP 2: Subtitle Loading Skipped")
			logger_config.info("No subtitle file provided, proceeding with caption-only matching")
		
		# Step 3: Smart frame extraction
		logger_config.info("STEP 3: Video Scene Detection and Frame Extraction")
		frame_extraction_start = time.time()
		frame_paths, frame_numbers, timestamps = self.extract_scenes_with_scenedetect(video_path)
		frame_extraction_time = time.time() - frame_extraction_start
		logger_config.info(f"Frame extraction completed in {frame_extraction_time:.2f} seconds")

		# Step 3.5: Pre-compute CLIP embeddings for all frames ---
		logger_config.info("STEP 3.5: Pre-computing CLIP embeddings for all frames")
		clip_embedding_start = time.time()
		frame_clip_embeddings = self.compute_frame_clip_embeddings(frame_paths)
		logger_config.info(f"CLIP embedding computation completed in {time.time() - clip_embedding_start:.2f} seconds")
		
		# Step 4: Enhanced captioning
		logger_config.info("STEP 4: Frame Caption Generation")
		captioning_start = time.time()
		captions = self.enhanced_caption_generation(frame_paths, timestamps)
		captioning_time = time.time() - captioning_start
		logger_config.info(f"Caption generation completed in {captioning_time:.2f} seconds")
		
		# Step 5: Process recap
		logger_config.info("STEP 5: Recap Text Processing and Sentence Segmentation")
		text_processing_start = time.time()
		sentences = self.split_recap_sentences(recap_text)
		text_processing_time = time.time() - text_processing_start
		logger_config.info(f"Text processing completed in {text_processing_time:.2f} seconds")
		
		# Step 6: Enhanced matching
		logger_config.info("STEP 6: Frame-Text Matching with Enhanced Scoring")
		
		results = []
		previous_timestamp = None
		self.load_clip()
		self.load_sentence_model()
		self.load_spacy()
		for i, sentence in enumerate(sentences):
			logger_config.info(f"Processing sentence {i+1} of {len(sentences)}")
			logger_config.debug(f"Sentence text: '{sentence}'")
			
			# Get enhanced matches
			sentence_matching_start = time.time()
			matches = self.find_best_match(
				captions, timestamps, sentence, frame_clip_embeddings, previous_timestamp, top_k=5
			)
			sentence_matching_time = time.time() - sentence_matching_start
			
			best_idx, best_score, score_breakdown = matches[0]
			timestamp = timestamps[best_idx]
			frame_num = frame_numbers[best_idx]
			
			# Get relevant subtitles for this timestamp
			relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=2.0)
			
			# Save frame
			output_path = os.path.join(TEMP_DIR, f"sentence_{i+1:02d}_frame_{frame_num}.jpg")
			import shutil
			shutil.copy2(frame_paths[best_idx], output_path)
			
			logger_config.info(f"Best match found: Frame {frame_num} at timestamp {timestamp:.1f}s")
			logger_config.info(f"Overall matching score: {best_score:.4f} (processed in {sentence_matching_time:.2f}s)")
			logger_config.debug(f"Score breakdown - Semantic: {score_breakdown['semantic']:.4f}, TF-IDF: {score_breakdown['tfidf']:.4f}, Subtitle: {score_breakdown['subtitle']:.4f}, Temporal: {score_breakdown['temporal']:.4f}")
			
			if relevant_subs:
				subtitle_texts = [sub['text'] for sub in relevant_subs[:2]]
				logger_config.info(f"Associated subtitles: {subtitle_texts}")
			else:
				logger_config.debug("No relevant subtitles found for this timestamp")
			
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
		self.unload_clip()
		self.unload_sentence_model()
		self.unload_spacy()
		# Step 6: Save enhanced results
		self.save_enhanced_results(results, OUTPUT_JSON)
		
		total_time = time.time() - overall_start_time
		avg_score = sum(r['score'] for r in results) / len(results) if results else 0

		logger_config.info("🎉 ENHANCED PROCESSING COMPLETE!")
		logger_config.info(f"📊 Total time: {total_time:.2f}s | Avg score: {avg_score:.4f}")
		
		return results

# Usage example
if __name__ == "__main__":
	# Initialize the enhanced matcher
	matcher = TextFrameAligner()
	
	# Your video and recap
	video_path = "input.mkv" 
	recap_text = """Radditz is dominating! — He stands over Goku, ready to strike the final blow, but Gohan can't stand by... He attacks with surprising force! — A power level over 1300... but it's not enough! Radditz easily deflects him... Goku knows he must act fast! — He pleads with Piccolo for assistance, and proposes a risky plan... Can they defeat Radditz together?— Piccolo agrees, but the Saiyan is too powerful... But they're buying time for Piccolo!…and with Raditz vulnerable Goku has an advantage!— Radditz is enraged by Goku's defiance, can they turn this battle?! With a final, desperate embrace, Goku sacrifices himself and, in a burst of pure power — Piccolo's Special Beam Cannon pierces through them both… But where does the story go from here?! — The countdown begins, for two even more powerful Saiyans in one year… Don't miss the next episode!!"""
	
	# Process with subtitle integration
	results = matcher.process(
		video_path=video_path,
		recap_text=recap_text,
		subtitle_path="subtitle.json"
	)
	
	logger_config.success(f"\n🎉 Enhanced processing complete! Check {OUTPUT_JSON} for detailed results.")