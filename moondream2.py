from vision_model import VisionModel
from PIL import Image
import time
import torch

class Moondream2(VisionModel):
	def load_model(self):
		from transformers import AutoModelForCausalLM, AutoTokenizer
		self.model = AutoModelForCausalLM.from_pretrained(
			"vikhyatk/moondream2",
			revision="2025-06-21",
			trust_remote_code=True,
			device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
		)

	def generate(self, image: Image.Image, text: str = "") -> str:
		with torch.inference_mode():
			return self.model.caption(image, length="normal")["caption"]

if __name__ == "__main__":
	model = Moondream2()
	image = Image.open("temp_dir/frames/scene_004_frame_935_at_frame_second39.00frame_second.jpg")
	text = None
	start_time = time.time()
	response = model.generate(image, text)
	end_time = time.time()
	print("Moondream2:", response)
	print(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")

# Captioning
# print("Short caption:")
# print(model.caption(image, length="short")["caption"])

# print("\nNormal caption:")
# for t in model.caption(image, length="normal", stream=True)["caption"]:
# 	# Streaming generation example, supported for caption() and detect()
# 	print(t, end="", flush=True)

# print(model.caption(image, length="normal")["caption"])

# Visual Querying
# print("\nVisual query: 'Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context.'")
# print(model.query(image, "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context.")["answer"])

# # Object Detection
# print("\nObject detection: 'face'")
# objects = model.detect(image, "face")["objects"]
# print(f"Found {len(objects)} face(s)")

# # Pointing
# print("\nPointing: 'person'")
# points = model.point(image, "person")["points"]
# print(f"Found {len(points)} person(s)")
