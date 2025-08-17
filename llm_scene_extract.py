import os
import subprocess
import cv2
from tqdm import tqdm
from custom_logger import logger_config

def run_transnetv2(video_path: str, frame_timestamps=None, start_from_sec=-1, end_from_sec=-1, skip_segment = [(None, None)]) -> list:
	transnetv2_dir = "/home/jebineinstein/git/TransNetV2"
	video_path = os.path.abspath(video_path)
	scene_txt_path = f"{video_path}.scenes.txt"

	# Step 1: Run TransNetV2 using subprocess inside venv
	activate_cmd = "source venv/bin/activate && python inference/transnetv2.py"
	cmd = f'{activate_cmd} "{video_path}"'
	subprocess.run(cmd, shell=True, check=True, cwd=transnetv2_dir, executable="/bin/bash")

	# Step 2: Verify .scenes.txt was created
	if not os.path.exists(scene_txt_path):
		raise FileNotFoundError(f"Scene file not found: {scene_txt_path}")
	print(f"📄 Found scene file: {scene_txt_path}")

	# Step 3: Read scenes
	with open(scene_txt_path, "r") as f:
		lines = f.readlines()
		scene_frames = [tuple(map(int, line.strip().split())) for line in lines if line.strip()]

	print(f"🎬 Detected {len(scene_frames)} scenes.")

	# Step 4: Convert frame numbers to seconds using cv2
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()
	if not fps:
		raise ValueError("Could not determine FPS from video.")

	scene_seconds = []
	for start, end in tqdm(scene_frames, desc="Converting frames to seconds"):
		start_sec = start / fps
		end_sec = end / fps

		# Apply user range filter
		if start_sec < start_from_sec or (end_from_sec != -1 and end_sec > end_from_sec):
			continue

		# Skip intro/outro (or other) segments if provided
		skip_flag = False
		for skip_start, skip_end in skip_segment:
			if skip_start is not None and skip_end is not None:
				# overlap check
				if not (end_sec < skip_start or start_sec > skip_end):
					skip_flag = True
					break
		if skip_flag:
			continue

		scene_seconds.append((start_sec, end_sec))

	if frame_timestamps:
		matched_scenes = []
		for ts in frame_timestamps:
			closest_scene = min(scene_seconds, key=lambda scene: min(abs(ts - scene[0]), abs(ts - scene[1])))
			matched_scenes.append(closest_scene)

		# Optional: remove duplicates if multiple timestamps matched same scene
		matched_scenes = list(set(matched_scenes))
	else:
		matched_scenes = scene_seconds

	logger_config.info("wait for memory to clear", seconds=5)
	return matched_scenes

# ✅ Example usage:
if __name__ == "__main__":
	scenes = run_transnetv2("input.mkv")
	print("\n🎯 Scene Time Ranges:")
	for start, end in scenes:
		print(f"{start:.2f}s - {end:.2f}s")
