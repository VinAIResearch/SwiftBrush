import argparse
import functools
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


def process_prompt_data(index, batch_start, prompt_embed, output_path):
    np.save(output_path / f"{batch_start+index}.npy", prompt_embed)
    return index


def wrapper_process_prompt_data(args):
    return process_prompt_data(*args)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

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

    return {"prompt_embeds": prompt_embeds.detach().cpu().numpy()}


def main(args):
    batch_size = args.batch_size
    num_processes = args.num_processes

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )

    text_encoder.requires_grad_(False)
    text_encoder.to("cuda", dtype=torch.float32)

    # Let's first compute all the embeddings so that we can free up the text encoders
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    with open(args.prompt_list) as f:
        prompts = f.read().splitlines()

    op = Path(args.prompt_list[:-4])
    op.mkdir(exist_ok=True, parents=True)

    if num_processes <= 1:
        for i in tqdm(range(len(prompts) // batch_size)):
            prompt_dicts = compute_embeddings_fn(prompts=prompts[batch_size * i : batch_size * i + batch_size])
            prompt_dicts = {key: value for key, value in prompt_dicts.items()}

            for j, prompt_embed in enumerate(prompt_dicts["prompt_embeds"]):
                np.save(op / f"{batch_size*i+j}.npy", prompt_embed)
    else:
        with Pool(num_processes) as pool:
            for i in tqdm(range(len(prompts) // batch_size), desc="Batch Progress"):
                prompt_dicts = compute_embeddings_fn(prompts=prompts[batch_size * i : batch_size * i + batch_size])
                prompt_dicts = {key: value for key, value in prompt_dicts.items()}

                batch_start = batch_size * i
                prompt_embeds = prompt_dicts["prompt_embeds"]

                _ = list(
                    tqdm(
                        pool.imap_unordered(
                            wrapper_process_prompt_data,
                            [
                                (
                                    j,
                                    batch_start,
                                    prompt_embeds[j],
                                    op,
                                )
                                for j in range(len(prompt_embeds))
                            ],
                        ),
                        total=len(prompt_embeds),
                        desc="Processing Prompts",
                        leave=False,
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a data preparation script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default=None,
        required=True,
        help="A .txt file containing all prompts used for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for encoding the text embedding.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes for encoding the text embedding.",
    )

    args = parser.parse_args()
    main(args)
