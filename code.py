import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import pipeline

classifier = pipeline(task="text-classification",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      device=device)
classifier("The concert was a breathtaking experience, and the musicians were \
# phenomenal.")

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    # "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    ).to(device)

    # We can also use other diffusion models, for example stable diffusion, hosted
# by RunwayML on HuggingFace
from diffusers import DiffusionPipeline

model_id = "segmind/tiny-sd"  # or any other compatible model
# model_id = "stabilityai/sdxl-turbo"
pipe = DiffusionPipeline.from_pretrained(model_id,
                                         dtype=torch.bfloat16, device_map="cuda")

prompt = "A goat in the center of a circular void of space, draped in a deep blue skin colour, with stars across his skin. The void is black, but he brings out the only bits of light throughout. His goat horns will extend upwards, crafted of bits of green folliage, with a yellow halo around him"
pipe(prompt).images[0]