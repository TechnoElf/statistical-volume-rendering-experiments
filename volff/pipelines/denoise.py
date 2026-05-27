import numpy as np
import torch

from volff.dataset import tile_image, untile_image
from volff.models.denoise import SimplePathTracerDenoiseModel
from volff.pipelines.isosurf import IsoSurfPipeline
from volff.pipelines.pathtrace import PathTracePipeline
from volff.pipelines.pipeline import Pipeline


class DenoisePipeline(Pipeline):
    def prepare(self, volume: np.ndarray, width: int = 1280, height: int = 720):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        self.volume = volume
        self.width = width
        self.height = height

        self.isosurf_pipeline = IsoSurfPipeline(self.ctx)
        self.isosurf_pipeline.prepare(volume, width, height)
        self.pathtrace_pipeline = PathTracePipeline(self.ctx)
        self.pathtrace_pipeline.prepare(volume, width, height)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimplePathTracerDenoiseModel()

    def render(self, params: dict):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        model_path = params.get("model_path", "model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        img_isosurf_1 = self.isosurf_pipeline.render({**params, "threshold": 0.10})
        img_isosurf_2 = self.isosurf_pipeline.render({**params, "threshold": 0.25})
        img_isosurf_6 = self.isosurf_pipeline.render({**params, "threshold": 0.65})
        img_isosurf_9 = self.isosurf_pipeline.render({**params, "threshold": 0.90})
        img_pathtrace = self.pathtrace_pipeline.render({**params, "iterations": 2})

        img_in_tiles = list(
            zip(
                *(
                    tile_image(img)
                    for img in (
                        img_pathtrace,
                        img_isosurf_1,
                        img_isosurf_2,
                        img_isosurf_6,
                        img_isosurf_9,
                    )
                )
            )
        )

        out_tiles = []
        for tile in img_in_tiles:
            in_imgs = []
            for img in (tile[0], tile[1], tile[2], tile[3], tile[4]):
                in_img = np.copy(img)
                in_img[:, :, 0:3] = in_img[:, :, 0:3] * 2 - 1
                in_img = torch.from_numpy(in_img).permute(2, 0, 1)
                in_imgs.append(in_img)

            in_tensor = torch.cat(in_imgs, dim=0).unsqueeze(0)
            with torch.no_grad():
                out_tensor = self.model(in_tensor.to(self.device))

            out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_tiles.append(out_img)

        img = untile_image(out_tiles, self.width, self.height)

        return img
