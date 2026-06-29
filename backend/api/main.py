import sys
sys.path.append(".")
import io, base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from pathlib import Path
from backend.models.model_loader import load_model, run_inference
from backend.explainability.attention_extractor import (
    extract_attention_maps, process_attention_for_image, create_heatmap_overlay
)

Path("outputs").mkdir(exist_ok=True)
app = FastAPI(title="VeriTrace AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
processor = None
device = None

def get_model():
    global model, processor, device
    if model is None:
        model, processor, device = load_model()
    return model, processor, device

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def bytes_to_image(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((336, 336))

def get_heatmap_images(m, inputs, image):
    try:
        attn_maps = extract_attention_maps(m, inputs)
        heatmap = process_attention_for_image(attn_maps, image.size)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap("jet")
        heatmap_colored = colormap(heatmap_norm)[:, :, :3]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
        overlay = create_heatmap_overlay(image, heatmap_norm)
        return image_to_base64(heatmap_img), image_to_base64(overlay)
    except Exception as e:
        print(f"Heatmap error: {e}")
        gray = np.array(image.convert("L"))
        gray_img = Image.fromarray(gray)
        return image_to_base64(gray_img), image_to_base64(image)

@app.get("/")
def root():
    return {"message": "VeriTrace AI running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), question: str = Form(default="What do you see?")):
    try:
        m, p, d = get_model()
        image = bytes_to_image(await file.read())
        answer = run_inference(m, p, d, image, question)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = p(text=[text], images=[image], return_tensors="pt")
        heatmap_b64, overlay_b64 = get_heatmap_images(m, inputs, image)
        return JSONResponse({"success": True, "question": question, "answer": answer,
                            "original_image": image_to_base64(image),
                            "heatmap_image": heatmap_b64,
                            "overlay_image": overlay_b64})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/counterfactual")
async def counterfactual(file: UploadFile = File(...), question: str = Form(default="What do you see?"),
                         x1: float = Form(default=0.25), y1: float = Form(default=0.25),
                         x2: float = Form(default=0.75), y2: float = Form(default=0.75),
                         mask_type: str = Form(default="black")):
    try:
        m, p, d = get_model()
        image = bytes_to_image(await file.read())
        original_answer = run_inference(m, p, d, image, question)
        masked = image.copy()
        w, h = masked.size
        ImageDraw.Draw(masked).rectangle([int(x1*w), int(y1*h), int(x2*w), int(y2*h)], fill=(0,0,0))
        masked_answer = run_inference(m, p, d, masked, question)
        return JSONResponse({"success": True, "question": question,
                            "original_answer": original_answer,
                            "masked_answer": masked_answer,
                            "answer_changed": original_answer != masked_answer,
                            "original_image": image_to_base64(image),
                            "masked_image": image_to_base64(masked)})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/prompt-variations")
async def prompt_variations(file: UploadFile = File(...),
                            questions: str = Form(default="What do you see?|Describe this image.")):
    try:
        m, p, d = get_model()
        image = bytes_to_image(await file.read())
        results = []
        for q in questions.split("|"):
            a = run_inference(m, p, d, image, q.strip())
            results.append({"question": q.strip(), "answer": a})
        return JSONResponse({"success": True, "results": results, "image": image_to_base64(image)})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)