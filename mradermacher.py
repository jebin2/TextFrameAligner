import base64
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
import os

def image_to_base64_data_uri(file_path):
    """
    Reads an image file and converts it to a base64 data URI.
    """
    with open(file_path, "rb") as img_file:
        return "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode('utf-8')

def main():
    # --- 1. Download the model and the multimodal projector file ---
    model_repo = "mradermacher/Qwen2.5-VL-7B-Abliterated-Caption-it-GGUF"
    model_filename = "Qwen2.5-VL-7B-Abliterated-Caption-it.Q4_K_M.gguf"
    mmproj_filename = "Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-f16.gguf" # This is the multi-modal projector file

    print(f"Downloading model: {model_filename} from {model_repo}")
    model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)

    print(f"Downloading multimodal projector: {mmproj_filename} from {model_repo}")
    mmproj_path = hf_hub_download(repo_id=model_repo, filename=mmproj_filename)

    # --- 2. Prepare the image ---
    # Make sure you have an image file named 'test_image.jpg' in the same directory
    # or provide a different path.
    image_path = "temp_dir/page-022.jpg_resize_frame__max_frame/sentence_12_frame_1.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Please place an image file named 'test_image.jpg' in the same directory as this script.")
        return

    image_uri = image_to_base64_data_uri(image_path)
    print("Image has been encoded to a data URI.")

    # --- 3. Initialize the model with the chat handler for vision models ---
    chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)

    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,  # Context window size
        n_gpu_layers=0,  # Set to 0 to run on CPU
        verbose=False # Set to True to see more detailed output
    )
    print("Model loaded successfully.")

    # --- 4. Create the prompt and generate a caption ---
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context."}
            ],
        }
    ]

    print("Generating caption...")
    response = llm.create_chat_completion(messages=messages)

    # --- 5. Print the generated caption ---
    if 'choices' in response and len(response['choices']) > 0:
        caption = response['choices'][0]['message']['content']
        print("\nGenerated Caption:")
        print(caption)
    else:
        print("\nFailed to generate a caption.")
        print("Response from model:", response)


if __name__ == "__main__":
    main()