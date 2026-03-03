import os
import sys
import requests
import json
import json_repair
from custom_logger import logger_config

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
PARENT_PAGE_ID = "3159a6c4-a97e-8097-8ef5-fd65ba5bfa07"

def get_headers():
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

def find_page_by_title(title):
    if not NOTION_API_KEY:
        logger.warning("NOTION_API_KEY not found in environment.")
        return None
        
    url = "https://api.notion.com/v1/search"
    payload = {
        "query": title,
        "filter": {
            "value": "page",
            "property": "object"
        }
    }
    logger.info(f"Calling Notion search API for: '{title}'")
    response = requests.post(url, headers=get_headers(), json=payload)
    if response.status_code == 200:
        results = response.json().get("results", [])
        logger.info(f"Notion search returned {len(results)} results")
        for page in results:
            page_title = ""
            properties = page.get("properties", {})
            title_prop = properties.get("title", {})
            title_arr = title_prop.get("title", [])
            if title_arr:
                page_title = title_arr[0].get("plain_text", "")
            if page_title == title:
                logger.info(f"✅ Exact title match found: '{page_title}'")
                return page
        logger.info(f"No exact title match among {len(results)} results")
    else:
        logger.error(f"Notion search API failed: {response.status_code} - {response.text}")
    return None

def check_for_result_in_notion(title):
    """
    Checks if a page exists in Notion under the parent page with the exact title.
    If it exists, looks for a code block containing JSON and parses it.
    """
    logger.info(f"Searching Notion for page with title: '{title}'")
    page = find_page_by_title(title)
    if not page:
        logger.info(f"No page found in Notion with title: '{title}'")
        return None
    
    page_id = page["id"]
    logger.info(f"Found Notion page: {page_id}, fetching blocks...")
    # Get blocks
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    response = requests.get(url, headers=get_headers())
    if response.status_code == 200:
        blocks = response.json().get("results", [])
        logger.info(f"Fetched {len(blocks)} blocks from Notion page")
        for i, block in enumerate(blocks):
            if block["type"] == "code":
                code_text = "".join([t["plain_text"] for t in block["code"]["rich_text"]])
                logger.info(f"Found code block (block {i+1}/{len(blocks)}), length: {len(code_text)} chars, attempting JSON parse...")
                try:
                    result = json_repair.loads(code_text)
                    if result:
                        logger.info(f"✅ Successfully parsed JSON from Notion code block ({len(result)} items)")
                        return result
                    else:
                        logger.info("Parsed JSON but result is empty/falsy")
                except Exception as e:
                    logger.warning(f"Found code block but couldn't parse JSON: {e}")
    else:
        logger.warning(f"Failed to fetch blocks from Notion: {response.status_code} - {response.text}")
    logger.info("No valid JSON result found in any Notion code block")
    return None

def split_text_into_chunks(text, max_length=2000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def update_or_create_notion_page(title, user_prompt, system_prompt):
    """
    Creates or updates a Notion page with the title and content.
    Used for appending the request details on failure.
    """
    if not NOTION_API_KEY:
        logger.warning("NOTION_API_KEY not found in environment.")
        return None
        
    page = find_page_by_title(title)
    
    def create_rich_text(content):
        return [{"type": "text", "text": {"content": chunk}} for chunk in split_text_into_chunks(content)]

    blocks = []
    
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "System Prompt"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": create_rich_text(f"System Prompt: {system_prompt}")
        }
    })
        
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "User Prompt"}}]
        }
    })

    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": create_rich_text(f"User Prompt: {user_prompt}")
        }
    })

    saved_response_code = "[\n  // paste JSON here\n]"

    if page:
        page_id = page["id"]
        url_get_blocks = f"https://api.notion.com/v1/blocks/{page_id}/children"
        res_blocks = requests.get(url_get_blocks, headers=get_headers())
        
        if res_blocks.status_code == 200:
            existing_blocks = res_blocks.json().get("results", [])
            has_response_heading = False
            for existing_block in existing_blocks:
                if existing_block["type"] == "heading_2":
                    rich_text = existing_block.get("heading_2", {}).get("rich_text", [])
                    if rich_text and "Response:" in rich_text[0].get("plain_text", ""):
                        has_response_heading = True
                elif existing_block["type"] == "code" and has_response_heading:
                    # Capture user's existing JSON response to prevent overwriting it
                    saved_response_code = "".join([t["plain_text"] for t in existing_block["code"]["rich_text"]])
                    break
            
            # Delete old blocks
            for existing_block in existing_blocks:
                block_id = existing_block["id"]
                url_delete = f"https://api.notion.com/v1/blocks/{block_id}"
                requests.delete(url_delete, headers=get_headers())

    # Construct the final blocks
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "Response:"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": create_rich_text(saved_response_code),
            "language": "json"
        }
    })

    if page:
        page_id = page["id"]
        # Append all new blocks
        url_patch = f"https://api.notion.com/v1/blocks/{page_id}/children"
        for i in range(0, len(blocks), 100):
            payload = {"children": blocks[i:i+100]}
            res = requests.patch(url_patch, headers=get_headers(), json=payload)
            if res.status_code != 200:
                logger.error(f"Failed to update Notion page: {res.text}")
    else:
        # Create new page
        url = "https://api.notion.com/v1/pages"
        payload = {
            "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            },
            "children": blocks[:100]
        }
        res = requests.post(url, headers=get_headers(), json=payload)
        if res.status_code != 200:
            logger.error(f"Failed to create Notion page: {res.text}")
            
        # Append remaining blocks if any
        if len(blocks) > 100 and res.status_code == 200:
            page_id = res.json()["id"]
            url = f"https://api.notion.com/v1/blocks/{page_id}/children"
            for i in range(100, len(blocks), 100):
                payload = {"children": blocks[i:i+100]}
                requests.patch(url, headers=get_headers(), json=payload)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    test_title = "test_cache_dir_1234"
    test_user_prompt = "[Scene 1, Scene 2]"
    test_system_prompt = "Match scenes to recap."
    
    # 1. Simulate failure -> Create page
    logger.info(f"Simulating failure, updating/creating page: {test_title}")
    update_or_create_notion_page(test_title, test_user_prompt, test_system_prompt)
    logger.info("Page should be created. Check Notion!")
    
    # 2. Simulate reading back (Wait for user to manually add JSON in Notion)
    logger.info(f"Let's see if we can find the page and parse any JSON: {test_title}")
    result = check_for_result_in_notion(test_title)
    if result:
        logger.info("Successfully read result from Notion:")
        print(json.dumps(result, indent=4))
    else:
        logger.info("No valid JSON result found in Notion yet (or page not found).")
