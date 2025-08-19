from browser_manager import BrowserManager
from browser_manager.browser_config import BrowserConfig
from custom_logger import logger_config
from gemiwrap.utils import compress_image
import os
import traceback

def get_caption_from_pally(user_prompt, file_path=None):
	try:
		config = BrowserConfig()
		config.docker_name = "pally_search_caption"

		with BrowserManager(config) as page:
			try:
				url = "https://pallyy.com/tools/image-description-generator"
				logger_config.info(f"Loading URL: {url}")
				page.goto(url)
				logger_config.info("Page loaded successfully, waiting 5s for content...")
				page.wait_for_timeout(5000)
				page.screenshot(path="get_caption_from_pally.png")

				if file_path:
					file_path = os.path.abspath(compress_image(file_path))
					logger_config.info(f"Uploading file: {file_path}")
					file_input = page.locator('input[type="file"]').first
					file_input.set_input_files(file_path)
					page.wait_for_selector('a[href="#remove"]', timeout=15000)
					logger_config.info("File uploaded successfully")
					page.wait_for_timeout(5000)
					page.screenshot(path="get_caption_from_pally.png")

				logger_config.info("Filling user prompt into input...")
				input = page.locator('input[name="description"]').first
				input.fill(user_prompt)
				logger_config.info("Prompt filled successfully")
				page.wait_for_timeout(2000)
				page.screenshot(path="get_caption_from_pally.png")

				logger_config.info("Clicking 'Convert Image to Description' button...")
				send_button = page.locator('button[type="submit"]').first
				send_button.click()
				logger_config.info("'Convert Image to Description' button clicked")
				page.wait_for_timeout(2000)
				page.screenshot(path="get_caption_from_pally.png")

				# Wait for the "Copy" button
				button = page.locator('button[type="button"]', has_text="Copy")
				button.wait_for(state="visible", timeout=15000)

				# Wait an extra 2 seconds (if needed)
				page.wait_for_timeout(2000)

				# Get the <p> tag immediately before the button
				p_locator = button.locator('xpath=preceding-sibling::p[1]')

				# Extract inner text
				result_text = p_locator.inner_text()

				print("Result:", result_text)

				page.screenshot(path="get_caption_from_pally.png")
				return result_text

			except Exception as e:
				logger_config.error(f"Error during get_caption_from_pally: {e} {traceback.format_exc()}")
				try:
					page.screenshot(path="get_caption_from_pally_error.png")
				except:
					pass
	except: pass

	return None

if __name__ == "__main__":
	get_caption_from_pally(
		user_prompt = (
			"Describe what is happening in this video frame as if you're telling a story. "
			"Focus on the main subjects, their actions, the setting, and any important "
			"details that would help someone understand the scene's context. "
			"Max word: 100 ONLY"
		),
		file_path = "/home/jebineinstein/git/CaptionCreator/media/movie_x_pic/Leaving Las Vegas 1995ANekPWfXeU_captioned_movie_review.png"
	)