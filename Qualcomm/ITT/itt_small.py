from geniex import AutoModelForCausalLM

MODEL = "unsloth/Qwen3-VL-4B-Instruct-GGUF"   # loads from C:\Users\shi-lab\.cache\geniex\models\...
IMAGE = r"C:\Users\shi-lab\AppData\Local\Programs\Benchsmart-Desktop-App\resources\app.asar.unpacked\genai_services\image_to_text\temp\data\Images\astronaut_horse.png"
DEVICE = "npu"   # try "npu", "hybrid", "gpu", "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL, precision="Q4_0", device_map=DEVICE)
messages = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE},
    {"type": "text", "text": "Describe the image."},
]}]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(prompt, images=[IMAGE], max_new_tokens=256)
print(out.text)
print(f"[{DEVICE}] {out.profile.generated_tokens} tok, {out.profile.decode_speed:.1f} tok/s")
model.close()
