import os
os.makedirs('backend/models', exist_ok=True)
lines = [
    'import torch\n',
    'from transformers import BlipProcessor, BlipForQuestionAnswering\n',
    'from PIL import Image\n',
    '\n',
    'def get_device():\n',
    '    return "cpu"\n',
    '\n',
    'def load_model():\n',
    '    print("Loading model...")\n',
    '    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")\n',
    '    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")\n',
    '    model.eval()\n',
    '    print("Model loaded!")\n',
    '    return model, processor, "cpu"\n',
    '\n',
    'def run_inference(model, processor, device, image, question):\n',
    '    inputs = processor(image, question, return_tensors="pt")\n',
    '    with torch.no_grad():\n',
    '        output = model.generate(**inputs, max_new_tokens=50)\n',
    '    return processor.decode(output[0], skip_special_tokens=True)\n',
]
with open('backend/models/model_loader.py', 'w') as f:
    f.writelines(lines)
print('Done!')