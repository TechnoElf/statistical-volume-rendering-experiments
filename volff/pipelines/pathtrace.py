import numpy as np
import slangpy as spy

from volff.pipelines.pipeline import Pipeline
from volff.transform import setup_transforms


class PathTracePipeline(Pipeline):
    def prepare(self, volume: np.ndarray, width: int = 1280, height: int = 720):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        self.volume = volume
        self.width = width
        self.height = height

        self.render_texture = self.ctx.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.unordered_access,
            label="render_texture",
        )

        self.profile = self.ctx.device.create_texture(
            type=spy.TextureType.texture_1d,
            format=spy.Format.r32_uint,
            width=64,
            usage=spy.TextureUsage.unordered_access,
            label="profile",
        )

        self.density_sampler = self.ctx.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
        )

        self.program = self.ctx.device.load_program("pathtracer.slang", ["main"])
        self.kernel = self.ctx.device.create_compute_kernel(self.program)

    def render(self, params: dict):
        if self.ctx is None:
            raise RuntimeError("Context not initialized")

        pitch = params.get("pitch", 0.0)
        yaw = params.get("yaw", 0.0)
        roll = params.get("roll", 0.0)
        iterations = params.get("iterations", 64)
        threshold = params.get("threshold", 0.8)
        scale = params.get("scale", 1.0)

        density_texture = self.ctx.device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=self.volume.shape[2],
            height=self.volume.shape[1],
            depth=self.volume.shape[0],
            usage=spy.TextureUsage.shader_resource,
            label="densities",
        )
        density_texture.copy_from_numpy(self.volume)

        light_dir = np.array([-0.5, 1.0, 1.0], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        light_color = np.array([10.0, 10.0, 10.0], dtype=np.float32)

        sigma_a = 1.0
        sigma_s = 100.0

        model, inv_model, view, inv_view, projection, inv_projection = setup_transforms(
            self.width, self.height, pitch, yaw, roll, scale
        )

        for i in range(iterations):
            self.kernel.dispatch(
                thread_count=[self.width, self.height, 1],
                vars={
                    "render_texture": self.render_texture,
                    "densities": density_texture,
                    "density_sampler": self.density_sampler,
                    "profile": self.profile,
                    "frame_index": i,
                    "model": np.array(model),
                    "inv_model": np.array(inv_model),
                    "view": np.array(view),
                    "inv_view": np.array(inv_view),
                    "projection": np.array(projection),
                    "inv_projection": np.array(inv_projection),
                    "light_dir": light_dir,
                    "light_color": light_color,
                    "sigma_a": sigma_a,
                    "sigma_s": sigma_s,
                    "threshold": threshold,
                },
            )

        img = self.render_texture.to_numpy()
        img = np.clip(img[..., :3] / (1.0 + img[..., :3]), 0, 1)
        img = np.dstack((img, np.ones((self.height, self.width), dtype=np.float32)))
        img[np.isnan(img)] = 0

        return img
