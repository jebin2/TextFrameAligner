import os
import json
from moondream2 import Moondream2
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from custom_logger import logger_config
from dotenv import load_dotenv
import os
if os.path.exists(".env"):
	print("Loaded load_dotenv")
	load_dotenv()

class MultiTypeCaptionGenerator:
	def __init__(self, cache_path, num_types=10, FYI="", local_only=False):
		self.cache_path = cache_path
		self.num_types = num_types
		self.lock = Lock()  # for safely updating temp JSON
		self.model = None
		self.FYI = FYI #FYI: This Movie Frame is from the movie called The Brides of dracula 1960
		self.local_only = local_only

	def _load_moondream2(self):
		if not self.model:
			self.model = Moondream2()

	def _load_temp(self, temp_path):
		if os.path.exists(temp_path):
			with open(temp_path, "r") as f:
				return json.load(f)
		return [{"in_progress": False, "processed": False, "caption": None, "dialogue": None} 
				for _ in range(self.num_frames)]

	def _save_temp(self, temp_path, data):
		with self.lock:
			with open(temp_path, "w") as f:
				json.dump(data, f, indent=4)

	def _get_next_index(self, temp_path):
		"""Get next available index and mark it as in_progress atomically"""
		with self.lock:
			# Always reload fresh data to avoid stale state
			temp_data = self._load_temp(temp_path)
			
			for i, entry in enumerate(temp_data):
				if not entry["in_progress"] and not entry["processed"]:
					temp_data[i]["in_progress"] = True
					# Save immediately to prevent other workers from picking same index
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
					print(f"🎯 Assigned index {i} to worker")
					return i, temp_data
			return None, temp_data

	def _worker(self, prompt, extract_scenes_json, temp_path, type_id):
		processed_count = 0
		print(f"🚀 Worker {type_id} started")
		
		while True:
			print(f"🔍 Worker {type_id} looking for next frame...")
			result_tuple = self._get_next_index(temp_path)
			if result_tuple[0] is None:
				print(f"✋ Worker {type_id} - no more frames available")
				break  # nothing left to process
			
			idx, temp_data = result_tuple
			print(f"📋 Worker {type_id} got index {idx}")
			
			scene = extract_scenes_json[idx]
			frame_path = scene["frame_path"][0]
			dialogue = scene["dialogue"]

			print(f"⚡ Type {type_id} processing frame {idx+1}/{len(extract_scenes_json)}")

			try:
				# Process the caption
				result = None
				if type_id % self.num_types == 0 or self.local_only:
					self._load_moondream2()
					result = self.model.generate(frame_path, prompt)
				else:
					new_prompt = f"""{prompt} Also identify all the characters name in this frame. Keep your description to exactly 100 words or fewer.
{self.FYI}"""
					result = self.search_in_ui_type(type_id, new_prompt, frame_path)

				print(f"📝 Type {type_id} got result for frame {idx}: {bool(result)}")

				# Update temp JSON with result
				with self.lock:
					# Reload to get fresh data before updating
					temp_data = self._load_temp(temp_path)
					
					if result:
						temp_data[idx]["caption"] = result.lower()
						temp_data[idx]["processed"] = True
						temp_data[idx]["dialogue"] = dialogue
						temp_data[idx]["frame_path"] = frame_path
						processed_count += 1
						print(f"✅ Type {type_id} completed frame {idx+1} (Total: {processed_count})")
					else:
						# Mark as failed but available for retry
						temp_data[idx]["processed"] = False
						print(f"❌ Type {type_id} failed frame {idx+1}")
					
					temp_data[idx]["in_progress"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
						
			except Exception as e:
				print(f"💥 Worker {type_id} error on frame {idx}: {e}")
				# Mark as failed and not in progress
				with self.lock:
					temp_data = self._load_temp(temp_path)
					temp_data[idx]["in_progress"] = False
					temp_data[idx]["processed"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
		
		print(f"🏁 Worker {type_id} finished. Processed {processed_count} frames.")

	def caption_generation(self, extract_scenes_json):
		self.num_frames = len(extract_scenes_json)
		cache_dir = os.path.join(self.cache_path, "caption_generation.json")
		partial_dir = os.path.join(self.cache_path, "partial_captions")
		os.makedirs(partial_dir, exist_ok=True)
		temp_path = os.path.join(partial_dir, "temp_progress.json")

		if os.path.exists(cache_dir):
			logger_config.info(f"✅ Using cached captions for {len(extract_scenes_json)} frames")
			with open(cache_dir, "r") as f:
				return json.load(f)

		logger_config.info(f"🚀 Starting multi-type caption generation for {len(extract_scenes_json)} frames")
		
		# Initialize temp file if it doesn't exist
		if not os.path.exists(temp_path):
			initial_data = [
				{
					"in_progress": False, 
					"processed": False, 
					"caption": None, 
					"dialogue": extract_scenes_json[i]["dialogue"],
					"frame_path": extract_scenes_json[i]["frame_path"][0]
				} 
				for i in range(self.num_frames)
			]
			self._save_temp(temp_path, initial_data)
		else:
			temp_data = self._load_temp(temp_path)
			
			# Handle case where temp file might have different number of frames
			if len(temp_data) != self.num_frames:
				logger_config.warning(f"⚠️  Temp file has {len(temp_data)} frames, expected {self.num_frames}. Reinitializing...")
				initial_data = [
					{
						"in_progress": False, 
						"processed": False, 
						"caption": None, 
						"dialogue": extract_scenes_json[i]["dialogue"],
						"frame_path": extract_scenes_json[i]["frame_path"][0]
					} 
					for i in range(self.num_frames)
				]
				self._save_temp(temp_path, initial_data)
			else:
				# Reset stuck frames and count existing progress
				completed_count = 0
				reset_count = 0
				
				for i, data in enumerate(temp_data):
					# Reset any frames that were in_progress (likely from interrupted run)
					if data["in_progress"]:
						data["in_progress"] = False
						data["processed"] = False
						data["caption"] = None
						reset_count += 1
					
					# Count completed frames
					if data["processed"] and data["caption"]:
						completed_count += 1
					
					# Ensure dialogue is set (in case structure changed)
					if i < len(extract_scenes_json):
						data["dialogue"] = extract_scenes_json[i]["dialogue"]
				
				self._save_temp(temp_path, temp_data)
				logger_config.info(f"📋 Resuming: {completed_count} completed, {reset_count} reset from in_progress")

		prompt = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details."

		# Start workers
		with ThreadPoolExecutor(max_workers=self.num_types) as executor:
			futures = []
			for type_id in range(self.num_types):
				future = executor.submit(self._worker, prompt, extract_scenes_json, temp_path, type_id)
				futures.append(future)
			
			# Wait for all workers to complete
			for future in futures:
				try:
					future.result()
				except Exception as e:
					logger_config.error(f"Worker failed with error: {e}")

		# Load final results and check for missing captions
		temp_data = self._load_temp(temp_path)
		captions = []

		for i, entry in enumerate(temp_data):
			captions.append({
				"scene_caption": entry["caption"], 
				"scene_dialogue": entry["dialogue"]
			})

		# Save final results
		with open(cache_dir, "w") as f:
			json.dump(captions, f, indent=4)
			
		logger_config.info(f"✅ Saved all captions to {cache_dir}")
		
		# Optionally clean up temp file
		# os.remove(temp_path)
		del self.model
		return captions

	def search_in_ui_type(self, type_id, prompt, file_path):
		from chat_bot_ui_handler import GoogleAISearchChat, AIStudioUIChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, GrokUIChat, MetaUIChat, CopilotUIChat, PallyUIChat
		from browser_manager.browser_config import BrowserConfig
		import os

		sources = [GoogleAISearchChat, AIStudioUIChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, GrokUIChat, MetaUIChat, CopilotUIChat, PallyUIChat]
		source = sources[type_id % self.num_types-1]

		try:
			config = BrowserConfig()
			config.starting_server_port_to_check = [9081, 10081, 11081, 12081, 13081, 14081, 15081, 16081, 17081][type_id % self.num_types-1]
			config.starting_debug_port_to_check = [10224, 11224, 12224, 13224, 14224, 15224, 16224, 17224, 18224][type_id % self.num_types-1]

			# Use a different profile for perplexity
			if source.__name__ == "GrokUIChat" or source.__name__ == "PerplexityUIChat":
				config.user_data_dir = os.getenv("PROFILE_PATH_1")
			else:
				config.user_data_dir = os.getenv("PROFILE_PATH")

			if source.__name__ == "MetaUIChat" or source.__name__ == "AIStudioUIChat":
				additional_flags = []
				additional_flags.append(f'-v {os.getcwd()}/{os.getenv("TEMP_OUTPUT", "chat_bot_ui_handler_logs")}:/home/neko/Downloads')
				additional_flags.append(f'-v /home/jebineinstein/git/browser_manager/policies.json:/etc/opt/chrome/policies/managed/policies.json')
				config.additionl_docker_flag = ' '.join(additional_flags)

			src_obj = source(config=config)
			result = src_obj.chat(user_prompt=prompt, file_path=file_path)
			if result and len(result.split(" ")) > 40:
				if "AI responses may include mistakes" in result:
					result = result[:result.index("AI responses may include mistakes")]

				if "Sources\nhelp" in result:
					result = result[:result.index("Sources\nhelp\n")]

				return result

		except Exception as e:
			logger_config.error(f"Type {type_id} failed processing: {e}")

		return None

# Usage example and main execution
if __name__ == "__main__":
	# Example usage
	captionGen = MultiTypeCaptionGenerator("temp_dir/cache_dir/3959d9724d997125d77addd3a7bf2738")
	
	try:
		# Process video-text alignment
		cache_dir = f"temp_dir/cache_dir/3959d9724d997125d77addd3a7bf2738/extract_scenes.json"

		with open(cache_dir, "r") as f:
			data = json.load(f)

		results = captionGen.caption_generation(data)
		
	except Exception as e:
		logger_config.error(f"Processing failed: {e}")
		raise