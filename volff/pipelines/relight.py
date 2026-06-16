import numpy as np
import torch
from PIL import Image

from volff.models import flux2_sampling
from volff.pipelines.levoy import LevoyPipeline
from volff.pipelines.pathtrace import PathTracePipeline
from volff.pipelines.pipeline import Pipeline


class RelightPipeline(Pipeline):
    def prepare(self, volume: np.ndarray, width: int = 1280, height: int = 720):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        self.volume = volume
        self.width = width
        self.height = height

        self.levoy_pipeline = LevoyPipeline(self.ctx)
        self.levoy_pipeline.prepare(volume, int(width / 2), int(height / 2))
        self.path_tracer_pipeline = PathTracePipeline(self.ctx)
        self.path_tracer_pipeline.prepare(volume, 1360, 768)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render(self, params: dict):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        ctx_path = params.get("ctx_path", "run/prompt_ctx_opt.pt")

        img_levoy = self.levoy_pipeline.render({**params})
        img_flux = flux2_sampling.gen(
            (img_levoy * 255).astype(np.uint8)[:, :, 0:3], ctx_path
        )

        return img_flux

    def train(self, params: dict):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        ctx_path = params.get("ctx_path", "run/prompt_ctx_opt.pt")

        img_levoy = self.levoy_pipeline.render({**params})
        img_path_tracer = self.path_tracer_pipeline.render({**params})

        flux2_sampling.train_ctx(
            (img_levoy * 255).astype(np.uint8)[:, :, 0:3],
            img_path_tracer[:, :, 0:3],
            num_iters=1,
            ctx_path=ctx_path,
        )
