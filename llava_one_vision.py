from vision_model import VisionModel
from PIL import Image
import torch
import time

class LlavaOneVision(VisionModel):
	def load_model(self):
		from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
		model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
		self.processor = AutoProcessor.from_pretrained(model_id)
		self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
			model_id, 
			torch_dtype=torch.float16, 
			low_cpu_mem_usage=True, 
		).to(self.device)

	def generate(self, image: Image.Image, text: str = "") -> str:
		with torch.inference_mode():
			conversation = [
				{

				"role": "user",
				"content": [
					{"type": "text", "text": text},
					{"type": "image"},
					],
				},
			]
			prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
			inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)
			output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
			result = self.processor.decode(output[0][2:], skip_special_tokens=True)
			if "assistant\n" in result:
				idx = result.find("assistant\n")
				result = result[idx + len("assistant\n"):].strip()
			return result

if __name__ == "__main__":
	model = LlavaOneVision()
	image = Image.open("temp_dir/frames/scene_0052_frame_3105_at_frame_second129.50frame_second.jpg")
	text = (
		"Describe what is happening in this video frame as if you're telling a story. "
		"Focus on the main subjects, their actions, the setting, and any important details "
		"that would help someone understand the scene's context."
	)

	start_time = time.time()
	response = model.generate(image, text)
	end_time = time.time()

	print("LlavaOneVision:", response)
	print(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")