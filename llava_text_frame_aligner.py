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
import traceback
import re # Added import for re
import shutil # Added import for shutil

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
TEMP_DIR = "temp_dir"
OUTPUT_JSON = f'{TEMP_DIR}/output.json'

class TextFrameAligner:
	def __init__(self, llava_model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", sentence_model_name='all-mpnet-base-v2', max_workers=None):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.max_workers = max_workers or max(8, mp.cpu_count()-4)
		logger_config.info("TextFrameAligner initialization started")
		logger_config.info(f"Compute device: {self.device}, Max workers: {self.max_workers}")

		# Environment setup with optimizations
		os.environ["HF_HUB_TIMEOUT"] = "120"
		if torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			torch.set_float32_matmul_precision('high')

		self.llava_model_name = llava_model_name
		self.sentence_model_name = sentence_model_name

		# Model containers
		self.llava_processor = None
		self.llava_model = None
		self.embedder = None
		self.nlp = None

		# Cached data
		self.subtitles = []
		self.subtitle_embeddings = None
		self._sentence_cache = {}
		self._llava_cache = {}

		# Updated weights - more emphasis on LLaVA's unified understanding
		self.weights = {
			'llava_direct': 0.50,  # Direct LLaVA text-image matching
			'llava_caption': 0.30,  # LLaVA caption semantic similarity
			'subtitle': 0.15,
			'temporal': 0.05
		}

		logger_config.info("TextFrameAligner initialization completed")

	@torch.inference_mode()
	def load_models_batch(self):
		"""Load LLaVA-OneVision and other models"""
		logger_config.info("Loading LLaVA-OneVision and supporting models")

		# Load LLaVA-OneVision
		from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

		self.llava_processor = LlavaOnevisionProcessor.from_pretrained(self.llava_model_name)
		self.llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
			self.llava_model_name,
			torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
			device_map="auto" if self.device == "cuda" else "cpu" # Changed to cpu if cuda not available
		)

		if self.device == "cuda" and hasattr(self.llava_model, 'to') and self.llava_model.device.type != "cuda":
			self.llava_model = self.llava_model.to(self.device)

		# Load SentenceTransformer for text embeddings
		self.embedder = SentenceTransformer(self.sentence_model_name)
		if self.device == "cuda":
			self.embedder = self.embedder.to(self.device)

		# Load spaCy
		import spacy
		try:
			self.nlp = spacy.load("en_core_web_sm")
		except OSError:
			from spacy.cli import download
			download("en_core_web_sm")
			self.nlp = spacy.load("en_core_web_sm")

		logger_config.info("All models loaded successfully")

	def extract_scenes(self, video_path: str, threshold: float = 30.0) -> Tuple[List[str], List[int], List[float]]:
		"""Scene extraction with parallel frame processing"""
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
				filename = f"scene_{i:03d}_frame_{frame_num}_at_frame_second{timestamp:.2f}frame_second.jpg"
				frame_path = os.path.join(frames_dir, filename)
				image.save(frame_path, format='JPEG', quality=85, optimize=True)
				logger_config.info(f"Save:: {frame_path}", overwrite=True)

			return frame_path, frame_num, timestamp

		# Use ThreadPoolExecutor for I/O bound operations
		frame_paths, frame_numbers, timestamps = [], [], []
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
	def llava_direct_matching(self, frame_paths: List[str], query_text: str, timestamps: List[float]) -> torch.Tensor:
		"""Direct text-image matching using LLaVA-OneVision"""
		logger_config.info(f"Computing LLaVA direct matching for query: {query_text[:50]}...")

		cache_key = f"direct_{hash(query_text)}"
		if cache_key in self._llava_cache:
			return self._llava_cache[cache_key]

		batch_size = 4
		all_scores = []

		# Create a matching prompt
		prompt_content = f"""You are evaluating how well this image matches the following text description: "{query_text}"

Please rate the match on a scale from 0 to 10, where:
- 0 means no match at all
- 10 means perfect match
- Consider visual elements, actions, emotions, and context

Respond with just the number (0-10)."""

		for i in range(0, len(frame_paths), batch_size):
			batch_paths = frame_paths[i:i + batch_size]
			batch_scores = []

			for frame_path in batch_paths:
				try:
					image = Image.open(frame_path).convert('RGB')

					# *** FIX: Use chat template to include <image> token ***
					messages = [{"role": "user", "content": f"<image>\n{prompt_content}"}]
					prompt_text = self.llava_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

					inputs = self.llava_processor(
						text=prompt_text,
						images=image,
						return_tensors="pt"
					).to(self.device)

					with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16 if self.device == "cuda" else torch.float32):
						outputs = self.llava_model.generate(
							**inputs,
							max_new_tokens=10,
							do_sample=False,
							pad_token_id=self.llava_processor.tokenizer.eos_token_id
						)

					# *** FIX: Decode only the newly generated tokens ***
					input_token_len = inputs.input_ids.shape[1]
					response = self.llava_processor.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0]
					
					score = self._extract_score_from_response(response)
					batch_scores.append(score)
					image.close()

				except Exception as e:
					logger_config.warning(f"LLaVA direct matching failed for {frame_path}: {e}")
					batch_scores.append(0.0)

			all_scores.extend(batch_scores)
			progress = min(100, (i + batch_size) / len(frame_paths) * 100)
			logger_config.info(f"LLaVA direct matching progress: {frame_path} {progress:.1f}%", overwrite=True)

		scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
		scores_tensor = scores_tensor / 10.0
		self._llava_cache[cache_key] = scores_tensor
		logger_config.info(f"LLaVA direct matching completed. Mean score: {scores_tensor.mean():.3f}")
		return scores_tensor

	def _extract_score_from_response(self, response: str) -> float:
		"""Extract numerical score from LLaVA response"""
		numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
		if numbers:
			try:
				score = float(numbers[-1])
				return max(0.0, min(10.0, score))
			except ValueError:
				pass

		response_lower = response.lower()
		if 'perfect' in response_lower or 'excellent' in response_lower: return 9.0
		elif 'good' in response_lower or 'well' in response_lower: return 7.0
		elif 'okay' in response_lower or 'decent' in response_lower: return 5.0
		elif 'poor' in response_lower or 'bad' in response_lower: return 2.0
		elif 'no' in response_lower or 'none' in response_lower: return 0.0
		return 3.0

	@torch.inference_mode()
	def llava_caption_generation(self, frame_paths: List[str], timestamps: List[float]) -> List[str]:
		"""Generate captions using LLaVA-OneVision with context"""
		logger_config.info(f"Starting LLaVA caption generation for {len(frame_paths)} frames")
		batch_size = 4
		captions = []
		
		base_prompt_content = """Describe this image in detail, focusing on:
- Actions and activities happening
- People and their expressions/emotions
- Objects and setting
- Mood and atmosphere
- Any text visible in the image

Provide a clear, detailed description in 1-2 sentences."""

		for i in range(0, len(frame_paths), batch_size):
			batch_paths = frame_paths[i:i + batch_size]
			batch_timestamps = timestamps[i:i + batch_size]
			batch_captions = []

			for j, (frame_path, timestamp) in enumerate(zip(batch_paths, batch_timestamps)):
				try:
					image = Image.open(frame_path).convert('RGB')

					relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=2.0)
					context = " ".join([sub['text'] for sub in relevant_subs[:2]])
					
					prompt_content = base_prompt_content
					if context.strip():
						prompt_content += f"\n\nContext from dialogue/narration: {context[:100]}"

					# *** FIX: Use chat template to include <image> token ***
					messages = [{"role": "user", "content": f"<image>\n{prompt_content}"}]
					prompt_text = self.llava_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
					
					inputs = self.llava_processor(
						text=prompt_text,
						images=image,
						return_tensors="pt"
					).to(self.device)

					with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16 if self.device == "cuda" else torch.float32):
						outputs = self.llava_model.generate(
							**inputs,
							max_new_tokens=100,
							temperature=0.3,
							do_sample=True,
							pad_token_id=self.llava_processor.tokenizer.eos_token_id
						)
					
					# *** FIX: Decode only the newly generated tokens ***
					input_token_len = inputs.input_ids.shape[1]
					caption = self.llava_processor.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0]
					
					batch_captions.append(caption or "A scene from the video")
					image.close()

				except Exception as e:
					logger_config.warning(f"LLaVA caption generation failed for {frame_path}: {e} {traceback.format_exc()}")
					batch_captions.append("A scene from the video")

			captions.extend(batch_captions)
			progress = min(100, (i + batch_size) / len(frame_paths) * 100)
			logger_config.info(f"LLaVA caption progress: {frame_path} {progress:.1f}%", overwrite=True)

		logger_config.info(f"LLaVA caption generation completed for {len(captions)} frames")
		return captions

	@lru_cache(maxsize=128)
	def get_cached_sentence_embeddings(self, text: str) -> torch.Tensor:
		"""Cache sentence embeddings to avoid recomputation"""
		return self.embedder.encode([text], convert_to_tensor=True)

	def load_subtitles(self, timestamp_data):
		"""Load subtitles and pre-compute embeddings"""
		logger_config.info(f"Loading timestamp_data")
		self.subtitles = timestamp_data
		if self.subtitles:
			subtitle_texts = [sub['text'] for sub in self.subtitles]
			self.subtitle_embeddings = self.embedder.encode(
				subtitle_texts,
				convert_to_tensor=True,
				batch_size=64
			)
			logger_config.info(f"Pre-computed embeddings for {len(subtitle_texts)} subtitles")

	def get_subtitles_for_timerange(self, start_time: float, end_time: float, buffer: float = 2.0) -> List[Dict]:
		"""Get subtitles for a time range"""
		if not self.subtitles:
			return []
		return [sub for sub in self.subtitles if sub['end'] >= start_time - buffer and sub['start'] <= end_time + buffer]

	def find_best_match(self, captions: List[str], timestamps: List[float], query: str, frame_paths: List[str], previous_timestamp: Optional[float] = None, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
		"""Find best matches using LLaVA-based scoring"""
		llava_direct_scores = self.llava_direct_matching(frame_paths, query, timestamps)
		query_emb = self.get_cached_sentence_embeddings(query)
		caption_embs = self.embedder.encode(captions, convert_to_tensor=True, batch_size=32)
		caption_similarities = util.cos_sim(query_emb, caption_embs)[0]

		subtitle_scores = np.zeros(len(timestamps))
		if self.subtitles and self.subtitle_embeddings is not None:
			query_sub_emb = self.get_cached_sentence_embeddings(query)
			for i, timestamp in enumerate(timestamps):
				relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=3.0)
				if relevant_subs:
					indices = [self.subtitles.index(sub) for sub in relevant_subs if sub in self.subtitles]
					if indices:
						relevant_embeddings = self.subtitle_embeddings[indices]
						similarities = util.cos_sim(query_sub_emb, relevant_embeddings)[0]
						subtitle_scores[i] = float(torch.max(similarities))

		temporal_scores = np.zeros(len(timestamps))
		if previous_timestamp is not None:
			time_diffs = np.array(timestamps) - previous_timestamp
			forward_mask = time_diffs > 0
			temporal_scores[forward_mask] = np.maximum(0, 1.0 - (time_diffs[forward_mask] / 30.0))
			backward_mask = time_diffs <= 0
			temporal_scores[backward_mask] = -0.5 * np.minimum(1.0, np.abs(time_diffs[backward_mask]) / 10.0)

		llava_direct_weighted = self.weights['llava_direct'] * llava_direct_scores.cpu().numpy()
		caption_weighted = self.weights['llava_caption'] * caption_similarities.cpu().numpy()
		subtitle_weighted = self.weights['subtitle'] * subtitle_scores
		temporal_weighted = self.weights['temporal'] * temporal_scores

		final_scores = llava_direct_weighted + caption_weighted + subtitle_weighted + temporal_weighted

		top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
		top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

		results = []
		for idx in top_indices:
			score_breakdown = {
				'llava_direct': float(llava_direct_scores[idx]),
				'caption_similarity': float(caption_similarities[idx]),
				'subtitle': subtitle_scores[idx],
				'temporal': temporal_scores[idx],
				'combined': float(final_scores[idx]) # Ensure it's a standard float
			}
			results.append((int(idx), float(final_scores[idx]), score_breakdown))
		return results

	def split_recap_sentences(self, text: str) -> List[str]:
		"""Split text into sentences using spaCy"""
		logger_config.info("Starting sentence splitting")
		doc = self.nlp(text)
		sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 8]
		logger_config.info(f"Generated {len(sentences)} sentences")
		return sentences

	def process(self, input_json_path: str):
		"""Main processing pipeline with LLaVA-OneVision"""
		logger_config.info("🚀 STARTING VIDEO-TEXT ALIGNMENT WITH LLAVA-ONEVISION")
		overall_start = time.time()

		self.reset()
		with open(input_json_path, 'r') as file:
			input_json = json.load(file)

		video_path = input_json["video_path"]
		recap_text = input_json["text"]
		timestamp_data = input_json["timestamp_data"]

		self.load_models_batch()
		if timestamp_data:
			self.load_subtitles(timestamp_data)

		frame_paths, frame_numbers, timestamps = self.extract_scenes(video_path)
		captions = self.llava_caption_generation(frame_paths, timestamps)
		sentences = self.split_recap_sentences(recap_text)

		results = []
		previous_timestamp = None
		for i, sentence in enumerate(sentences):
			matches = self.find_best_match(
				captions, timestamps, sentence, frame_paths,
				previous_timestamp, top_k=5
			)
			if not matches:
				logger_config.warning(f"No match found for sentence: {sentence}")
				continue

			best_idx, best_score, score_breakdown = matches[0]
			timestamp = timestamps[best_idx]
			frame_num = frame_numbers[best_idx]

			frame_second_str = frame_paths[best_idx].split("frame_second")[1].split("frame_second")[0]
			output_path = os.path.join(TEMP_DIR, f"sentence_{i+1:02d}_frame_{frame_num}_frame_second{frame_second_str}frame_second.jpg")
			shutil.copy2(frame_paths[best_idx], output_path)

			relevant_subs = self.get_subtitles_for_timerange(timestamp, timestamp, buffer=2.0)
			results.append({
				'sentence_id': i + 1, 'sentence': sentence, 'frame_number': frame_num,
				'timestamp': timestamp, 'caption': captions[best_idx], 'score': best_score,
				'score_breakdown': score_breakdown, 'output_path': output_path,
				'relevant_subtitles': relevant_subs
			})
			previous_timestamp = timestamp
			logger_config.info(f"Processed sentence {i+1}/{len(sentences)} - Score: {best_score:.4f}", overwrite=True)

		self.save_enhanced_results(results, OUTPUT_JSON)
		total_time = time.time() - overall_start
		avg_score = sum(r['score'] for r in results) / len(results) if results else 0

		logger_config.info(f"🎉 PROCESSING COMPLETE WITH LLAVA-ONEVISION!")
		logger_config.info(f"⚡ Total time: {total_time:.2f}s | Avg score: {avg_score:.4f}")
		return results

	def reset(self):
		"""Reset with better memory management"""
		logger_config.info("Starting reset")
		self._sentence_cache.clear()
		self._llava_cache.clear()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()

		gc.collect()

		if os.path.exists(TEMP_DIR):
			shutil.rmtree(TEMP_DIR)
		os.makedirs(TEMP_DIR, exist_ok=True)
		logger_config.info("Reset completed")

	def save_enhanced_results(self, results: List[dict], output_path: str):
		"""Save results to JSON file"""
		logger_config.info(f"Saving results to: {output_path}")
		json_output = []
		for result in results:
			frame_second_str = result['output_path'].split("frame_second")[1].split("frame_second")[0]
			json_entry = {
				"frame_path": os.path.abspath(result['output_path']),
				"frame_second": float(frame_second_str), # Convert to float
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


# Usage example with LLaVA-OneVision
if __name__ == "__main__":
	matcher = TextFrameAligner(
		llava_model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
		max_workers=6
	)
	results = matcher.process("input.json")