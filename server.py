from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
from typing import List, Optional

app = FastAPI()

# Serve static files (your HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

class FrameData(BaseModel):
    scene_start: float
    scene_end: float
    best_time: float
    frame_path: List[str]
    dialogues: List[str]
    dialogue: str
    characters_in_frame: Optional[str] = ""

class UpdateCharactersRequest(BaseModel):
    index: int
    characters: str

# Global variable to store current JSON file path
current_json_path = None

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

class LoadJsonRequest(BaseModel):
    file_path: str

@app.post("/load-json")
async def load_json(request: LoadJsonRequest):
    """Load JSON file from the provided path"""
    global current_json_path
    
    file_path = request.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON file not found")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        current_json_path = file_path
        return {"data": data, "message": "JSON loaded successfully"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/update-characters")
async def update_characters(request: UpdateCharactersRequest):
    """Update characters for a specific frame and save to file"""
    global current_json_path
    
    if not current_json_path:
        raise HTTPException(status_code=400, detail="No JSON file loaded")
    
    try:
        # Read current data
        with open(current_json_path, 'r') as f:
            data = json.load(f)
        
        # Update the specific frame
        if 0 <= request.index < len(data):
            data[request.index]['characters_in_frame'] = request.characters
            
            # Save back to file
            with open(current_json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return {"message": "Characters updated and saved successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid frame index")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating file: {str(e)}")

@app.get("/current-data")
async def get_current_data():
    """Get current JSON data"""
    global current_json_path
    
    if not current_json_path:
        raise HTTPException(status_code=400, detail="No JSON file loaded")
    
    try:
        with open(current_json_path, 'r') as f:
            data = json.load(f)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.get("/image")
async def download_annotations(file_path: str):

    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=os.path.basename(file_path)
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Frame Character Editor Server...")
    print("📁 Make sure to create a 'static' folder and put your HTML file there as 'index.html'")
    print("🌐 Server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)