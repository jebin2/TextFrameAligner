from custom_logger import logger_config
import cv2
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
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
	def __init__(self, blip_model="Salesforce/blip-image-captioning-large", sentence_model='all-mpnet-base-v2'):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logger_config.info("TextFrameAligner initialization started")
		logger_config.info(f"Compute device selected: {self.device}")
		
		# Environment setup
		os.environ["TORCH_USE_CUDA_DSA"] = "1"
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		os.environ["HF_HUB_TIMEOUT"] = "120"
		logger_config.debug("CUDA environment variables configured")

		self.blip_model = blip_model
		self.sentence_model = sentence_model
		self.subtitles = []
		self.subtitle_embeddings = None
		self._load_model()
		
		# Improved matching weights
		self.weights = {
			'semantic': 0.35,
			'tfidf': 0.20,
			'entity': 0.15,
			'subtitle': 0.25,  # New subtitle matching
			'temporal': 0.05   # Temporal coherence bonus
		}
		logger_config.info("Matching weights configured: semantic=0.35, tfidf=0.20, entity=0.15, subtitle=0.25, temporal=0.05")
		
		logger_config.info("TextFrameAligner initialization completed successfully")

	def _load_model(self):
		"""Load all required models with detailed logging."""
		try:
			logger_config.info("Starting model loading process")
			
			# Load BLIP model
			logger_config.info(f"Loading BLIP image captioning model: {self.blip_model}")
			start_time = time.time()
			
			from transformers import BlipProcessor, BlipForConditionalGeneration
			logger_config.debug("Importing BLIP components from transformers library")
			
			logger_config.info("Loading BLIP processor")
			self.processor = BlipProcessor.from_pretrained(self.blip_model)
			
			logger_config.info("Loading BLIP conditional generation model")
			self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model).to(self.device)
			
			blip_load_time = time.time() - start_time
			logger_config.info(f"BLIP model loading completed in {blip_load_time:.2f} seconds")
			
			# Load sentence transformer
			logger_config.info(f"Loading sentence transformer model: {self.sentence_model}")
			start_time = time.time()
			
			self.embedder = SentenceTransformer(self.sentence_model)
			
			sentence_load_time = time.time() - start_time
			logger_config.info(f"Sentence transformer loading completed in {sentence_load_time:.2f} seconds")

			# Load spaCy
			self._load_spacy()
			
		except Exception as e:
			logger_config.error(f"Model loading failed with error: {str(e)}")
			raise RuntimeError(f"Failed to load model: {str(e)}")

	def _load_spacy(self):
		"""Load spaCy model with error handling and logging"""
		try:
			logger_config.info("Loading spaCy NLP model: en_core_web_sm")
			start_time = time.time()
			
			self.nlp = spacy.load("en_core_web_sm")
			
			spacy_load_time = time.time() - start_time
			logger_config.info(f"spaCy model loading completed in {spacy_load_time:.2f} seconds")
			
		except OSError:
			logger_config.warning("spaCy model not found locally, initiating download")
			from spacy.cli import download
			logger_config.info("Downloading en_core_web_sm model")
			download("en_core_web_sm")
			self.nlp = spacy.load("en_core_web_sm")
			logger_config.info("spaCy model download and loading completed")

	def load_subtitles(self, subtitle_path: str):
		"""Load and process subtitle data for enhanced matching"""
		logger_config.info(f"Loading subtitle data from: {subtitle_path}")

		with open(subtitle_path, 'r', encoding='utf-8') as f:
			self.subtitles = json.load(f)

		logger_config.info(f"Loaded {len(self.subtitles)} subtitle entries from file")
		
		# Create subtitle text corpus for embedding
		subtitle_texts = [sub['text'] for sub in self.subtitles]
		
		if subtitle_texts:
			logger_config.info("Computing sentence embeddings for subtitle corpus")
			self.subtitle_embeddings = self.embedder.encode(subtitle_texts, convert_to_tensor=True)
			logger_config.info(f"Generated embeddings for {len(subtitle_texts)} subtitle entries")
		
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
		
		logger_config.debug(f"Found {len(relevant_subs)} subtitles overlapping with timerange {start_time:.1f}-{end_time:.1f}s (buffer: {buffer}s)")
		return relevant_subs

	def extract_scenes_with_scenedetect(self, video_path: str, threshold: float = 30.0) -> Tuple[List[str], List[int], List[float]]:
		"""Extract one frame per scene using modern PySceneDetect API."""
		from scenedetect import detect, ContentDetector
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
			
			if (i + 1) % 20 == 0:
				logger_config.info(f"Frame extraction progress: {i + 1}/{len(scene_list)} scenes processed", overwrite=True)

		cap.release()
		logger_config.info(f"Frame extraction completed. Extracted {len(frame_paths)} frames from {len(scene_list)} scenes")
		return frame_paths, frame_numbers, timestamps

	def enhanced_caption_generation(self, frame_paths: List[str], timestamps: List[float]) -> List[str]:
		"""Enhanced captioning that incorporates subtitle context"""
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
				
				with torch.no_grad():
					out = self.blip_model.generate(
						**inputs, 
						max_length=100,  # Increased for richer descriptions
						num_beams=8,
						length_penalty=0.8,
						repetition_penalty=1.2,
						do_sample=True,
						temperature=0.7
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
				logger_config.error(f"Caption generation failed for frame {frame_path}: {str(e)}")
				captions.append("Error processing frame")
		
		logger_config.info(f"Caption generation completed for all {len(captions)} frames")
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
			logger_config.debug(f"Forward temporal progression: {previous_timestamp:.1f}s -> {current_timestamp:.1f}s (diff: {time_diff:.1f}s, score: {score:.4f})")
		else:
			# Backward jump penalty
			score = -0.5 * min(1.0, time_diff / 10.0)
			logger_config.debug(f"Backward temporal jump: {previous_timestamp:.1f}s -> {current_timestamp:.1f}s (diff: {time_diff:.1f}s, penalty: {score:.4f})")
		
		return score

	def find_best_match_enhanced(self, captions: List[str], timestamps: List[float], query: str, 
							   previous_timestamp: Optional[float] = None, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
		"""Enhanced matching with subtitle integration and temporal coherence"""
		logger_config.debug(f"Starting enhanced matching for query: '{query[:100]}...'")
		
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
		logger_config.debug("Computing entity matching scores")
		entity_scores = self.enhance_matching_with_entities(captions, query)
		
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
				self.weights['semantic'] * float(semantic_similarities[i]) +
				self.weights['tfidf'] * float(tfidf_similarities[i]) +
				self.weights['entity'] * entity_scores[i] +
				self.weights['subtitle'] * subtitle_scores[i] +
				self.weights['temporal'] * temporal_scores[i]
			)
			
			score_breakdown = {
				'semantic': float(semantic_similarities[i]),
				'tfidf': float(tfidf_similarities[i]),
				'entity': entity_scores[i],
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

	def enhance_matching_with_entities(self, captions: List[str], query: str) -> List[float]:
		"""Enhanced entity matching with character names and anime-specific terms"""
		logger_config.debug("Starting entity extraction from query text")
		doc = self.nlp(query)
		
		# Extract entities with expanded categories
		entities = []
		for ent in doc.ents:
			if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']:
				entities.append(ent.text.lower())
		
		logger_config.debug(f"Extracted {len(entities)} named entities: {entities}")
		
		# Extract important nouns and proper nouns
		important_terms = []
		for token in doc:
			if (token.pos_ in ['NOUN', 'PROPN'] and 
				not token.is_stop and 
				len(token.text) > 2):
				important_terms.append(token.text.lower())
		
		logger_config.debug(f"Extracted {len(important_terms)} important terms: {important_terms[:10]}...")
		
		# Add anime/action specific terms
		action_terms = ['attack', 'power', 'energy', 'beam', 'blast', 'fight', 'battle', 
					   'strike', 'punch', 'kick', 'transformation', 'aura']
		query_lower = query.lower()
		action_terms_found = []
		for term in action_terms:
			if term in query_lower:
				important_terms.append(term)
				action_terms_found.append(term)
		
		if action_terms_found:
			logger_config.debug(f"Found action-specific terms: {action_terms_found}")
		
		key_terms = list(set(entities + important_terms))
		logger_config.debug(f"Total key terms for matching: {len(key_terms)}")
		
		if not key_terms:
			logger_config.debug("No key terms found, returning zero scores for all captions")
			return [0.0] * len(captions)
		
		# Calculate fuzzy matching scores
		logger_config.debug("Computing entity matching scores with fuzzy matching")
		keyword_scores = []
		for i, caption in enumerate(captions):
			caption_lower = caption.lower()
			exact_matches = sum(1 for term in key_terms if term in caption_lower)
			
			# Add fuzzy matching for character names
			fuzzy_matches = 0
			for term in key_terms:
				if len(term) > 4:  # Only for longer terms
					matches = difflib.get_close_matches(term, caption_lower.split(), n=1, cutoff=0.8)
					if matches:
						fuzzy_matches += 0.5  # Half score for fuzzy matches
			
			total_score = exact_matches + fuzzy_matches
			keyword_scores.append(total_score)
			
			if total_score > 0:
				logger_config.debug(f"Caption {i}: {exact_matches} exact + {fuzzy_matches:.1f} fuzzy matches = {total_score:.1f}")
		
		# Normalize scores
		max_score = max(keyword_scores) if keyword_scores else 1
		normalized_scores = [score / max_score if max_score > 0 else 0 for score in keyword_scores]
		
		logger_config.debug(f"Entity matching completed. Max raw score: {max_score}, normalized scores computed")
		return normalized_scores

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
		
		doc = self.nlp(cleaned_text)
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
		matching_start = time.time()
		
		results = []
		previous_timestamp = None
		
		for i, sentence in enumerate(sentences):
			logger_config.info(f"Processing sentence {i+1} of {len(sentences)}")
			logger_config.debug(f"Sentence text: '{sentence}'")
			
			# Get enhanced matches
			sentence_matching_start = time.time()
			matches = self.find_best_match_enhanced(
				captions, timestamps, sentence, previous_timestamp, top_k=5
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
			logger_config.debug(f"Score breakdown - Semantic: {score_breakdown['semantic']:.4f}, TF-IDF: {score_breakdown['tfidf']:.4f}, Entity: {score_breakdown['entity']:.4f}, Subtitle: {score_breakdown['subtitle']:.4f}, Temporal: {score_breakdown['temporal']:.4f}")
			
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