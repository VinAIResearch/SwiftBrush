import argparse

import torch
from diffusers import DDPMScheduler, DiffusionPipeline
from torchvision.utils import save_image


@torch.no_grad()
def encode_prompt(pipe, prompt):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    captions = [prompt]
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(pipe, encode_func, prompt, generator, device, weight_dtype):
    vae = pipe.vae

    prompt_embed = encode_func(pipe, prompt)["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)

    input_shape = (prompt_embed.shape[0], 4, 64, 64)
    input_noise = torch.randn(input_shape, generator=generator, device=device, dtype=weight_dtype)

    pred_original_sample = predict_original(pipe, input_noise, prompt_embed)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=weight_dtype)).sample.float()
    return (image + 1) / 2


def predict_original(pipe, input_noise, prompt_embeds):
    unet = pipe.unet
    scheduler = pipe.scheduler

    max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
    max_timesteps = max_timesteps * (scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds).sample

    latents = (input_noise - sigma_T * model_pred) / alpha_T
    return latents


def main(args):
    device, dtype = "cuda", torch.float16

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, scheduler=scheduler, torch_dtype=dtype
    )
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = inference(pipe, encode_prompt, args.prompt, generator, device, dtype)
    save_image(image[0], "result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="thuanz123/swiftbrush",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Text prompt used for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
        help="Random seed used for inference.",
    )

    args = parser.parse_args()
    main(args)
