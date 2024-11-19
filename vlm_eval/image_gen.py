import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from PIL import Image
import open_clip
import numpy as np
from tqdm import tqdm
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_clip_model():
    clip, _, processor = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device
    )
    # checkpoint = torch.load('C:/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/simclip4.pt', map_location=device)
    # clip.visual.load_state_dict(checkpoint)
    return clip

def loss_fn(x, clip, prompt):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer(prompt).to(device)
    image_features = clip.encode_image(x)
    text_features = clip.encode_text(text)
    loss = F.cosine_similarity(image_features, text_features)
    return loss

class CosineSchedule:
    def __init__(self, num_steps: int, phase_coef: float = 0.9):
        self.num_steps = num_steps
        self.max_phase = phase_coef * np.pi / 2

    def __call__(self, iter: int):
        s = np.cos(iter / self.num_steps * self.max_phase)
        return s

def main(args):
    clip = load_clip_model()
    img_size = 112
    downsample = v2.Resize((img_size, img_size), antialias=True)
    upsample = v2.Resize((224, 224), antialias=True)

    scheduler = CosineSchedule(num_steps=args.steps)
    batch_size = 2
    x = torch.randn(batch_size, 3, 8, 8).to(device)

    x[0, 0, :, :] *= 0.26862954
    x[0, 1, :, :] *= 0.26130258
    x[0, 2, :, :] *= 0.27577711

    x[0, 0, :, :] += 0.48145466
    x[0, 1, :, :] += 0.4578275
    x[0, 2, :, :] += 0.40821073

    x = x.requires_grad_()

    for i in tqdm(range(args.steps)):
        x = x.detach().requires_grad_()
        x = upsample(x)
        loss = loss_fn(x, clip, args.prompt)
        loss = list(loss)
        if i % args.verbose == 0:
            print([elem.item() for elem in loss])

        grad = torch.autograd.grad(loss, x)
        grad = torch.cat(grad)
        grad = grad / torch.linalg.vector_norm(grad, dim=[1,2,3]).view(-1, 1, 1, 1)
        s = scheduler(i)
        x = x + img_size * s * grad
        x = x.clamp(-1, 1)
        x = downsample(x)

    images = upsample(x).detach().cpu()
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]

    # Save images
    os.makedirs('image_gen', exist_ok=True)
    pil_images[0].save(f'image_gen/generated_image_1_steps_{args.steps}.png')
    pil_images[1].save(f'image_gen/generated_image_2_steps_{args.steps}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Generation Script")
    parser.add_argument('--steps', type=int, default=300, help='Number of steps for the generation process')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for the image generation')
    parser.add_argument('--verbose', type=int, default=5, help='Verbosity level for logging')
    args = parser.parse_args()
    main(args)