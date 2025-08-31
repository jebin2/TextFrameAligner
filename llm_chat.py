from custom_logger import logger_config
import custom_env
import requests
import subprocess
import time
import os
import common

class ChatService:

    def __init__(self, model = "gemma3:latest", unload=False, from_online=False):
        if common.get_device() == "cpu":
            os.environ["OLLAMA_NO_GPU"] = "1"
        else:
            os.environ.pop("OLLAMA_NO_GPU", None)

        import ollama
        self.client = ollama.Client(host=custom_env.OLLAMA_REQ_URL)
        self.default_model = model
        self.unload = unload
        self.from_online = from_online
        
        # Initialize Ollama and model
        self._ensure_ollama_running()
        self._pull_model()
        
        # Quick health check
        if not self._is_ollama_running():
            raise RuntimeError("Ollama is not running. Please start it with: ollama serve")

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{custom_env.OLLAMA_REQ_URL}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False

    def _start_ollama(self) -> bool:
        """Start Ollama service"""
        try:
            logger_config.info("Starting Ollama service...")
            # Try to start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Wait for service to start (max 30 seconds)
            for _ in range(30):
                time.sleep(1)
                if self._is_ollama_running():
                    logger_config.info("Ollama service started successfully")
                    return True
            
            logger_config.error("Ollama service failed to start within 30 seconds")
            return False
            
        except FileNotFoundError:
            logger_config.error("Ollama not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger_config.error(f"Error starting Ollama: {e}")
            return False

    def _ensure_ollama_running(self):
        """Ensure Ollama service is running, start if necessary"""
        if not self._is_ollama_running():
            logger_config.warning("Ollama is not running. Attempting to start...")
            if not self._start_ollama():
                raise RuntimeError("Failed to start Ollama service")

    def _pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            logger_config.info(f"Pulling model {self.default_model}...")
            # Pull model with progress tracking
            for progress in self.client.pull(self.default_model, stream=True):
                if 'status' in progress:
                    print(f"\r{progress['status']}", end="", flush=True)
            print()  # New line after progress
            logger_config.info(f"Model {self.default_model} pulled successfully")
            return True
        except Exception as e:
            logger_config.error(f"Error pulling model {self.default_model}: {e}")
            return False

    def _build_message(self, user_prompt, system_message=None, history=None, images=None):
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        if history:
            messages.extend([{"role": msg['role'], "content": msg['content']} for msg in history])

        if images:
            messages.append({"role": "user", "content": user_prompt, "images": [images]})
        else:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    def generate_response(self, user_prompt, system_message=None, history=None, images=None, format=None, num_ctx=8100):
        try:
            if self.from_online:
                return self.get_gemini_response(user_prompt, system_message, history, images, format)

            logger_config.debug(f"System Prompt: {system_message}")
            logger_config.debug(f"User Prompt: {user_prompt}")
            logger_config.debug(f"History: {history}")
            logger_config.debug(f"images: {images}")

            messages = self._build_message(user_prompt, system_message, history, images)

            response = self.client.chat(
                model=self.default_model,
                messages=messages,
                stream=True,
                format=format,
                options={
                    "num_ctx": num_ctx
                }
            )

            content = ""
            for chunk in response:
                current_data = chunk.get('message', {}).get('content')
                current_data = current_data.replace('\\"', '')
                content += current_data
                print(current_data, end="", flush=True)

            # content = response.get('message', {}).get('content')
            logger_config.debug(f"Content: {content}")
            if self.unload:
                self.unload_modal()

            return content

        except Exception as e:
            raise ValueError(f"Error during chat response generation: {e}")

    def get_gemini_response(self, user_prompt, system_message=None, history=None, images=None, format=None):
        from gemini_config import pre_model_wrapper
        geminiWrapper = pre_model_wrapper(
            system_instruction=system_message,
            schema=format,
            history=history
        )
        model_responses = geminiWrapper.send_message(
            user_prompt=user_prompt
        )
        return model_responses

    def unload_modal(self):
        """Unload the currently loaded model using Ollama CLI"""
        try:
            import subprocess

            logger_config.info(f"Unloading model {self.default_model}...")
            subprocess.run(
                ["ollama", "stop", self.default_model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            logger_config.info(f"Model {self.default_model} unloaded successfully.", seconds=5)
        except Exception as e:
            logger_config.error(f"Failed to unload model {self.default_model}: {e}")

    def __enter__(self):
        """Support with-statement context manager"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Automatically unload the model on context exit"""
        self.unload_modal()
        # Don't suppress exceptions
        return False

    def __del__(self):
        """Destructor: unload model when object is destroyed"""
        try:
            self.unload_modal()
        except Exception:
            # Avoid exceptions in __del__
            pass

if __name__ == "__main__":
    from text_content import TextContent
    from anime_review import AnimeReview
    from language_translator import Translator
    from manga_reviewer import MangaReview

    content = MangaReview(None, False)
    chat_service = ChatService()

    response = chat_service.generate_response(
        user_prompt=content.get_user_prompt(),
        images=content.get_images_prompt(),
        system_message=content.get_system_prompt(),
        history=None,
        format=content.json_schema()
    )
    chat_service.unload_modal()
    # logger_config_config.debug(response)
    # joined_explanation = " ".join([moment["explain"] for moment in response])
    # logger_config_config.debug(joined_explanation)