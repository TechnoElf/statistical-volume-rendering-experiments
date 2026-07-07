import random

import deepspeed
import numpy as np
import torch
from accelerate import cpu_offload
from deepspeed.ops.adam import DeepSpeedCPUAdam
from einops import rearrange
from PIL import Image
from torch import nn

from volff.models.flux2 import Klein4BParams
from volff.models.flux2_sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    encode_image_refs,
    get_schedule,
    load_ae,
    load_flow_model,
    scatter_ids,
)

config = {
    "repo_id": "black-forest-labs/FLUX.2-klein-4B",
    "ae_repo_id": "black-forest-labs/FLUX.2-dev",
    "filename": "flux-2-klein-4b.safetensors",
    "filename_ae": "ae.safetensors",
    "params": Klein4BParams(),
    "model_path": "KLEIN_4B_MODEL_PATH",
    "defaults": {"guidance": 1.0, "num_steps": 4},
    "fixed_params": {
        "guidance",
        "num_steps",
    },  # guidance and timestep distilled
    "guidance_distilled": True,
}


class Flux2Gen(nn.Module):
    def __init__(self, torch_device, seed):
        super().__init__()

        self.torch_device = torch_device
        self.width, self.height = 1360, 768
        self.seed = seed

        self.model = load_flow_model(config, device=torch_device)
        self.ae = load_ae(config, device=torch_device)

    def forward(self, img, ctx):
        ref_tokens, ref_ids = encode_image_refs(self.ae, [img])

        ctx, ctx_ids = batched_prc_txt(ctx)

        # Create noise
        shape = (1, 128, self.height // 16, self.width // 16)
        generator = torch.Generator(device=self.torch_device).manual_seed(self.seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device=self.torch_device
        )
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device=self.torch_device
        )
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(config["defaults"]["num_steps"], x.shape[1])

        x = denoise(
            self.model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=config["defaults"]["guidance"],
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = self.ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        x = 0.5 * (x + 1.0)

        return x


def flux2_gen(
    image: np.ndarray,
    ctx_path: str = "run/prompt_ctx_opt.pt",
) -> np.ndarray:
    torch_device = torch.device("cuda")

    ctx = torch.load(ctx_path)
    ctx = ctx.to(device="cuda")
    img = Image.fromarray(image)

    model = Flux2Gen(torch_device, random.randrange(2**31))
    model.eval()
    model = cpu_offload(model, torch_device)
    with torch.no_grad():
        x = model(img, ctx)

    return x.cpu().numpy()


class Flux2TrainContext(nn.Module):
    def __init__(self, torch_device, seed, ctx):
        super().__init__()

        self.torch_device = torch_device
        self.width, self.height = 1360, 768
        self.seed = seed
        self.ctx = torch.nn.Parameter(ctx.clone().float())

        self.model = load_flow_model(config, device=torch_device)
        self.ae = load_ae(config, device=torch_device)

        # Freeze model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.ae.parameters():
            p.requires_grad_(False)

    def forward(self, img, target):
        ref_tokens, ref_ids = encode_image_refs(self.ae, [img])

        ctx, ctx_ids = batched_prc_txt(self.ctx.bfloat16())

        # Create noise
        shape = (1, 128, self.height // 16, self.width // 16)
        generator = torch.Generator(device=self.torch_device).manual_seed(self.seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device=self.torch_device
        )
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device=self.torch_device
        )
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(config["defaults"]["num_steps"], x.shape[1])

        x = denoise(
            self.model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=config["defaults"]["guidance"],
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = self.ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = 0.5 * (x[0] + 1.0)

        loss = torch.nn.functional.mse_loss(x, target)

        return loss


def flux2_train_context(
    image: np.ndarray,
    target: np.ndarray,
    ctx_path: str = "run/prompt_ctx_opt.pt",
):
    torch_device = torch.device("cuda")

    ctx = torch.load(ctx_path).to(device="cuda")
    img = Image.fromarray(image)
    target_t = torch.from_numpy(target).float().cuda().permute(2, 0, 1)

    model = Flux2TrainContext(torch_device, random.randrange(2**31), ctx)
    model.eval()
    # model = cpu_offload(model, torch_device)
    # optimizer = DeepSpeedCPUAdam([model.ctx], lr=1e-3)
    optimizer = torch.optim.Adam([model.ctx], lr=1e-3)
    # model_engine, optimizer, _, _ = deepspeed.initialize(
    #     config={
    #         "train_batch_size": 1,
    #         "zero_optimization": {
    #             "stage": 3,
    #             "contiguous_gradients": True,
    #             "stage3_max_live_parameters": 1e9,
    #             "stage3_max_reuse_distance": 1e9,
    #             "stage3_prefetch_bucket_size": 1e7,
    #             "stage3_param_persistence_threshold": 1e5,
    #             "reduce_bucket_size": 5e8,
    #             "sub_group_size": 1e9,
    #             "offload_optimizer": {"device": "cpu"},
    #             "offload_param": {"device": "cpu"},
    #         },
    #         "zero_force_ds_cpu_optimizer": False,
    #     },
    #     model=model,
    #     optimizer=optimizer,
    # )

    optimizer.zero_grad()

    loss = model_engine(img, target_t)
    # model_engine.backward(loss)
    # model_engine.step()
    loss = model(img, target_t)
    loss.backward()
    optimizer.step()

    ctx_param = model.ctx_param.detach().cpu().bfloat16()
    torch.save(ctx_param, ctx_path)
