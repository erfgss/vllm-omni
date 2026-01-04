# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time
import inspect
from pathlib import Path
from typing import Any, Iterable, Tuple, Dict, List,Union, Callable

import torch
from torch import nn
from PIL import Image
import numpy as np

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging as hf_logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor

from vllm.logger import init_logger
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .transformer_flux import FluxTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL



if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


hf_logger = hf_logging.get_logger(__name__)   # diffusers / HF
logger = init_logger(__name__)                # vLLM


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Union[int, None] = None,
    device: Union[Union[str, torch.device], None] = None,
    timesteps: Union[List[int], None] = None,
    sigmas: Union[List[float], None] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 32
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64

        print(self.vae.config)
        print(self.transformer.config)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Union[torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Union[torch.device, None] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Union[torch.device, None] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Union[torch.FloatTensor, None] = None,
        pooled_prompt_embeds: Union[torch.FloatTensor, None] = None,
        max_sequence_length: int = 512,
        lora_scale: Union[float, None] = None,
    ):

        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[Union[str, List[str]], None] = None,
        height: Union[int, None] = None,
        width: Union[int, None] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Union[int, None] = 1,
        generator: Union[torch.Generator, List[torch.Generator], None] = None,
        latents: Union[torch.FloatTensor, None] = None,
        prompt_embeds: Union[torch.FloatTensor, None] = None,
        pooled_prompt_embeds: Union[torch.FloatTensor, None] = None,
        output_type: Union[str, None] = "pil",
        return_dict: bool = True,
        partitioned: bool = True,
        joint_attention_kwargs: Union[Dict[str, Any], None] = None,
        callback_on_step_end:Union[Callable[[int, int, Dict], None], None] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None

                noise_pred = self.transformer(
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            # Support both custom AutoencoderKL (with `partitioned`) and diffusers' AutoencoderKL
            decode_kwargs = {"return_dict": False}
            try:
                sig = inspect.signature(self.vae.decode)
                if "partitioned" in sig.parameters:
                    decode_kwargs["partitioned"] = partitioned
                elif partitioned:
                    logger.warning("partitioned=True requested but VAE.decode has no 'partitioned' arg; using standard decode.")
            except (ValueError, TypeError):
                # If signature inspection fails, fall back to safest call without partitioned
                pass
            decoded = self.vae.decode(latents, **decode_kwargs)
            image = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)



_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _normalize_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.bfloat16
    k = str(dtype).lower()
    if k not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported: {sorted(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[k]


def _device_from_config(_: OmniDiffusionConfig) -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_images(
    images: list[Image.Image],
    out_dir: str | Path,
    basename: str,
    ext: str = "png",
) -> list[str]:
    out_dir = _ensure_dir(out_dir)
    paths: list[str] = []
    for i, im in enumerate(images):
        fp = out_dir / f"{basename}_{i:02d}.{ext}"
        im.save(fp)
        paths.append(str(fp))
    return paths


class UltraFluxPipeline(nn.Module):
    """
    vLLM-Omni native UltraFlux diffusion pipeline.

    IMPORTANT:
    ◦ FluxPipeline may NOT be an nn.Module and may NOT implement state_dict().

    ◦ vLLM DiffusersLoader still expects model.load_weights(weights_iter).

    ◦ Therefore we MUST load weights into submodules (vae/transformer/encoders) individually.

    """

    # Common component names you might have in UltraFlux/Flux-style pipelines.
    # We will probe these on self.pipe and load weights by prefix matching.
    _COMPONENT_ATTRS: tuple[str, ...] = (
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",      # not a module, ignored
        "tokenizer_2",    # not a module, ignored
        "image_encoder",
        "unet",           # if present
    )

    def __init__(self, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config

        model_id = getattr(od_config, "model", None)
        if not model_id:
            raise ValueError("OmniDiffusionConfig.model must be set for UltraFluxPipeline.")

        # device & dtype
        self.device: torch.device = _device_from_config(od_config)
        self.torch_dtype: torch.dtype = _normalize_dtype(getattr(od_config, "dtype", "bf16"))

        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=self.torch_dtype,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )


        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype
        )

        tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=self.torch_dtype
        )

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipe = FluxPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 4

        # FluxPipeline might implement .to(), but it may only move internal modules.
        self.pipe = pipe
        if hasattr(self.pipe, "to"):
            self.pipe = self.pipe.to(self.device)

        # vllm-omni expects these in initialize_model()
        self.vae = getattr(self.pipe, "vae", None)
        self.transformer = getattr(self.pipe, "transformer", None)

        # tokenizer max length
        for tok_name in ("tokenizer", "tokenizer_2"):
            tok = getattr(self.pipe, tok_name, None)
            if tok is not None and hasattr(tok, "model_max_length"):
                tok.model_max_length = 512

        self._apply_vae_optimizations()
        self.eval()

    def _iter_submodules(self) -> List[tuple[str, nn.Module]]:
        mods: List[tuple[str, nn.Module]] = []
        # Prefer explicit known attributes on pipe
        for attr in self._COMPONENT_ATTRS:
            obj = getattr(self.pipe, attr, None)
            if isinstance(obj, nn.Module):
                mods.append((attr, obj))
        # Also include any nn.Modules registered directly on self (unlikely, but safe)
        for name, obj in super().named_children():
            if isinstance(obj, nn.Module):
                mods.append((name, obj))
        # De-duplicate by id
        seen = set()
        uniq: List[tuple[str, nn.Module]] = []
        for n, m in mods:
            if id(m) in seen:
                continue
            seen.add(id(m))
            uniq.append((n, m))
        return uniq

    def parameters(self, recurse: bool = True):
        # vLLM frequently calls model.parameters()
        for _, m in self._iter_submodules():
            yield from m.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, m in self._iter_submodules():
            pfx = f"{prefix}.{name}" if prefix else name
            yield from m.named_parameters(prefix=pfx, recurse=recurse)


    def load_weights(self, weights_iter: Iterable[Tuple[str, torch.Tensor]]):
        """
        vLLM loader yields (name, tensor) pairs.

        Because FluxPipeline may not be an nn.Module (no state_dict),
        we dispatch incoming weights into component modules based on prefix:
            transformer.xxx -> self.pipe.transformer.load_state_dict({xxx: ...})
            vae.xxx         -> self.pipe.vae.load_state_dict({xxx: ...})
            text_encoder.xxx, text_encoder_2.xxx, etc.

        Returns:
            set of loaded fully-qualified keys for diagnostics.
        """
        # Build module map from pipeline components
        module_map: Dict[str, nn.Module] = {}
        for attr, mod in self._iter_submodules():
            module_map[attr] = mod

        # Accumulate per-module tensors with stripped prefix
        bucket: Dict[str, Dict[str, torch.Tensor]] = {k: {} for k in module_map.keys()}
        loaded: set[str] = set()

        # Prefixes sometimes used by loaders
        strip_prefixes = ("pipe.", "model.", "module.", "")

        for raw_name, tensor in weights_iter:
            if tensor is None:
                continue

            # Normalize name by stripping common leading prefixes
            name = raw_name
            for pfx in strip_prefixes:
                if pfx and name.startswith(pfx):
                    name = name[len(pfx):]
                    break

            # Expect form "component.subkey..."
            if "." not in name:
                # Not a component param, skip
                continue

            comp, subkey = name.split(".", 1)

            # Some loaders may use "vae_decoder"/"vae_encoder" patterns; best-effort remap
            if comp not in module_map:
                if comp.startswith("vae") and "vae" in module_map:
                    comp = "vae"
                elif comp.startswith("transformer") and "transformer" in module_map:
                    comp = "transformer"
                elif comp.startswith("text_encoder_2") and "text_encoder_2" in module_map:
                    comp = "text_encoder_2"
                elif comp.startswith("text_encoder") and "text_encoder" in module_map:
                    comp = "text_encoder"
                else:
                    continue

            bucket[comp][subkey] = tensor
            loaded.add(f"{comp}.{subkey}")

        # Load each module (non-strict to tolerate buffers/extras)
        for comp, mod in module_map.items():
            sd_part = bucket.get(comp, {})
            if not sd_part:
                continue
            missing, unexpected = mod.load_state_dict(sd_part, strict=False)
            if missing:
                logger.debug("UltraFlux load_weights: %s missing=%d (first=%s)", comp, len(missing), missing[:5])
            if unexpected:
                logger.debug("UltraFlux load_weights: %s unexpected=%d (first=%s)", comp, len(unexpected), unexpected[:5])

        return loaded


    def _apply_vae_optimizations(self):
        vae = getattr(self, "vae", None)
        if vae is None:
            return

        use_slicing = bool(getattr(self.od_config, "vae_use_slicing", False))
        use_tiling = bool(getattr(self.od_config, "vae_use_tiling", False))

        if hasattr(vae, "use_slicing"):
            try:
                vae.use_slicing = use_slicing
            except Exception:
                pass

        if hasattr(vae, "use_tiling"):
            try:
                vae.use_tiling = use_tiling
            except Exception:
                pass

        # UltraFlux partitioned decode (4K/8K critical) if available
        if hasattr(vae, "decode"):
            try:
                sig = inspect.signature(vae.decode)
                if hasattr(vae, "config"):
                    if "partitioned" in sig.parameters:
                        setattr(vae.config, "use_partitioned_decode", use_tiling)
                    else:
                        setattr(vae.config, "use_partitioned_decode", False)
            except Exception:
                pass

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 4.0,
        max_sequence_length: int | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> DiffusionOutput:
        t0 = time.perf_counter()

        p = getattr(req, "prompt", None) if req is not None else None
        p = p if p is not None else prompt
        if p is None:
            return DiffusionOutput(error="Prompt is required.")

        h = (getattr(req, "height", None) or height or 1024)
        w = (getattr(req, "width", None) or width or 1024)
        steps = (getattr(req, "num_inference_steps", None) or num_inference_steps or 50)

        cfg = getattr(req, "guidance_scale", None)
        cfg = cfg if cfg is not None else guidance_scale

        max_len = (getattr(req, "max_sequence_length", None) or max_sequence_length or 512)

        num_images_per_prompt = int(
            getattr(req, "num_outputs_per_prompt", None)
            or getattr(req, "batch_size", None)
            or 1
        )

        if isinstance(p, str):
            prompts = [p]
        else:
            prompts = list(p)

        prompts_expanded: list[str] = []
        for pp in prompts:
            prompts_expanded.extend([pp] * num_images_per_prompt)

        seed = getattr(req, "seed", None)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        device_type = self.device.type
        autocast_dtype = self.torch_dtype if device_type == "cuda" else None

        # Optional offline saving (default OFF; only if explicitly provided)
        save_dir = kwargs.get("save_dir") or getattr(req, "save_dir", None)
        save_dir = str(save_dir) if save_dir else None
        save_name = kwargs.get("save_name") or getattr(req, "save_name", None) or "ultraflux"
        save_ext = kwargs.get("save_ext") or getattr(req, "save_ext", None) or "png"


        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    out = self.pipe(
                        prompts_expanded,
                        height=int(h),
                        width=int(w),
                        num_inference_steps=int(steps),
                        guidance_scale=float(cfg),
                        max_sequence_length=int(max_len),
                        generator=generator,
                )
           
        images: list[Image.Image] = list(out.images)

        if save_dir:
                _save_images(images, save_dir, str(save_name), str(save_ext))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
                "UltraFlux done: batch=%d steps=%d cfg=%.2f size=%dx%d time=%.1fms",
                len(prompts_expanded),
                int(steps),
                float(cfg),
                int(w),
                int(h),
                dt_ms,
            )

        return DiffusionOutput(output=images)
