from vision_model import VisionModel
from PIL import Image
import torch
import time
import common

MID = "apple/FastVLM-1.5B"
IMAGE_TOKEN_INDEX = -200  # what the model code looks for

class AppleFastVLM(VisionModel):
	def load_model(self):
		from transformers import AutoTokenizer, AutoModelForCausalLM
		self.tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained(
			MID,
			dtype=torch.float16 if common.is_gpu_available() else torch.float32,
			device_map="auto",
			trust_remote_code=True
		)

	def generate(self, image: Image.Image, text: str = "") -> str:
		messages = [
			{"role": "user", "content": f"<image>\n{text}"}
		]
		rendered = self.tokenizer.apply_chat_template(
			messages, add_generation_prompt=True, tokenize=False
		)
		pre, post = rendered.split("<image>", 1)
		# Tokenize the text *around* the image token (no extra specials!)
		pre_ids  = self.tokenizer(pre,  return_tensors="pt", add_special_tokens=False).input_ids
		post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
		# Splice in the IMAGE token id (-200) at the placeholder position
		img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
		input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
		attention_mask = torch.ones_like(input_ids, device=self.model.device)
		# Preprocess image via the model's own processor
		px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
		px = px.to(self.model.device, dtype=self.model.dtype)
		# Generate
		with torch.inference_mode():
			out = self.model.generate(
				inputs=input_ids,
				attention_mask=attention_mask,
				images=px,
				max_new_tokens=128,
			)
		return self.tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
	model = AppleFastVLM()
	image = Image.open("/home/jebin/git/TextFrameAligner/temp_dir/Blue is the Warmest Color 2013_2161_4322/frames/scene_0000_at_9.38s.jpg")
	text = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context."

	start_time = time.time()
	response = model.generate(image, text)
	end_time = time.time()

	print("AppleFastVLM:", response)
	print(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")
