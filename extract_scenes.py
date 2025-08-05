import os
import json
from custom_logger import logger_config
from typing import List, Tuple
from scenedetect import detect, ContentDetector
import cv2
from PIL import Image
import numpy as np

def get_segments(video_path: str) -> List[dict]:
	output_json = '/home/jebineinstein/git/STT/stt/temp_dir/output_transcription.json'
	with open(output_json, 'r', encoding='utf-8') as file:
		result = json.load(file)

	return result["segments"]["segment"]

def is_mostly_black_fast(image: Image.Image, black_threshold=20, percentage_threshold=0.9):
		"""Fast black frame detection"""
		width, height = image.size
		if width > 200 or height > 200:
			scale = min(200/width, 200/height)
			new_size = (int(width * scale), int(height * scale))
			image = image.resize(new_size, Image.LANCZOS)
		
		grayscale_image = image.convert('L')
		pixels = np.array(grayscale_image)
		
		black_pixel_count = np.sum(pixels < black_threshold)
		total_pixels = pixels.size
		
		black_percentage = black_pixel_count / total_pixels
		return black_percentage >= percentage_threshold

def map_dialogues_to_scenes(scene_list: List[Tuple[float, float]], dialogues: List[dict], video_path: str, frames_dir: str) -> List[dict]:
	"""
	Map each dialogue to its corresponding scene and save a frame from each scene.
	"""
	os.makedirs(frames_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)

	frames_extracted = 0

	scene_dialogue_map = []

	for i, (scene_start, scene_end) in enumerate(scene_list):
		# Get middle frame timestamp
		mid_time = (scene_start + scene_end) / 2.0
		mid_frame = int(mid_time * fps)

		cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
		ret, frame = cap.read()

		frame_filename = f"scene_{frames_extracted:04d}_at_{scene_start:.2f}s.jpg"
		frame_path = os.path.join(frames_dir, frame_filename)

		if ret:
			cv2.imwrite(frame_path, frame)
			with Image.open(frame_path) as image:
				if is_mostly_black_fast(image):
					logger_config.warning(f"Skipping black frame at {scene_start:.2f}s")
					os.remove(frame_path)
					frame_path = None
				else:
					frames_extracted += 1
		else:
			frame_path = None
			print(f"[WARN] Failed to extract frame for scene {i}")

		if frame_path:
			scene_dialogues = [
				d for d in dialogues
				if d['end'] >= scene_start and d['start'] <= scene_end
			]

			scene_dialogue_map.append({
				"scene_start": scene_start,
				"scene_end": scene_end,
				"frame_path": frame_path,
				"dialogues": scene_dialogues
			})

	cap.release()
	return scene_dialogue_map

def combine_consecutive_same_dialogues(scene_map: List[dict]) -> List[dict]:
	if not scene_map:
		return []

	def are_dialogues_equal(d1, d2):
		return json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

	combined = []
	current = scene_map[0].copy()
	if not isinstance(current["frame_path"], list):
		current["frame_path"] = [current["frame_path"]]

	for next_scene in scene_map[1:]:
		next_dialogues = next_scene["dialogues"]
		current_dialogues = current["dialogues"]

		if are_dialogues_equal(current_dialogues, next_dialogues):
			# Extend scene range and append frame
			current["scene_end"] = next_scene["scene_end"]
			current["frame_path"].append(next_scene["frame_path"])
		else:
			combined.append(current)
			current = next_scene.copy()
			if not isinstance(current["frame_path"], list):
				current["frame_path"] = [current["frame_path"]]

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
	scene_list = detect(
		video_path,
		ContentDetector(threshold=threshold),
		show_progress=True
	)

	all_scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

	if frame_timestamps:
		matched_scenes = []
		for ts in frame_timestamps:
			closest_scene = min(all_scenes, key=lambda scene: min(abs(ts - scene[0]), abs(ts - scene[1])))
			matched_scenes.append(closest_scene)

		# Optional: remove duplicates if multiple timestamps matched same scene
		matched_scenes = list(set(matched_scenes))
	else:
		matched_scenes = all_scenes

	return matched_scenes

def extract_scenes(video_path: str, frame_timestamp: List[float], cache_path, frames_dir, threshold: float = 30.0):
	cache_dir = f"{cache_path}/extract_scenes.json"

	# If cache exists and frame_timestamp is empty, use cached scene detection
	if os.path.exists(cache_dir) and not frame_timestamp:
		logger_config.info(f"Using cached scene detection: {video_path}")
		with open(cache_dir, "r") as f:
			data = json.load(f)
		return data

	print("[INFO] Detecting scenes...")
	scenes = detect_scenes(video_path, frame_timestamp, threshold)

	print("[INFO] Extracting dialogues...")
	dialogues = get_segments(video_path)

	print("[INFO] Mapping dialogues to scenes and extracting frames...")
	scene_dialogue_map = map_dialogues_to_scenes(scenes, dialogues, video_path, frames_dir)

	print("[INFO] Combining consecutive scenes with identical dialogues...")
	scene_dialogue_map = combine_consecutive_same_dialogues(scene_dialogue_map)
	scene_dialogue_map = combine_dialogues(scene_dialogue_map)

	# Save as JSON
	with open(cache_dir, "w", encoding="utf-8") as f:
		json.dump(scene_dialogue_map, f, indent=2)
	print(f"\n[INFO] Mapping saved to {cache_dir}")

	return scene_dialogue_map

if __name__ == "__main__":
	video_path = "reuse/019_-_Defying_Gravity/019_-_Defying_Gravity_compressed.mp4"
	extract_scenes(video_path)
