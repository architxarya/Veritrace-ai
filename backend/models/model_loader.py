import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def get_device():
    return "cpu"

def load_model():
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.to("cpu")
    model.eval()
    print("Model loaded!")
    return model, processor, "cpu"

def run_inference(model, processor, device, image, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = [output_ids[i][len(inputs["input_ids"][i]):] for i in range(len(output_ids))]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]