import importlib.resources as resources
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import slangpy as spy

import volff
from volff.transform import setup_transforms


class Tracer:
    def __init__(self, kernel_path, width, height):
        self.width = width
        self.height = height

        self.device = spy.create_device(
            type=spy.DeviceType.vulkan,
            include_paths=[kernel_path],
        )

        self.render_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.unordered_access,
            label="render_texture",
        )

        self.profile = self.device.create_texture(
            type=spy.TextureType.texture_1d,
            format=spy.Format.r32_uint,
            width=64,
            usage=spy.TextureUsage.unordered_access,
            label="profile",
        )

        self.density_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
        )

        self.pt_program = self.device.load_program("pathtracer.slang", ["main"])
        self.pt_kernel = self.device.create_compute_kernel(self.pt_program)

        self.rc_program = self.device.load_program("raycaster.slang", ["main"])
        self.rc_kernel = self.device.create_compute_kernel(self.rc_program)

    @classmethod
    @contextmanager
    def create(cls, width, height):
        with ExitStack() as stack:
            kernel_path = stack.enter_context(resources.path(volff, "kernels"))
            yield cls(kernel_path, width, height)

    def trace(self, volume, iterations=128, pitch=0.0, yaw=0.0, roll=0.0):
        density_texture = self.device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=volume.shape[2],
            height=volume.shape[1],
            depth=volume.shape[0],
            usage=spy.TextureUsage.shader_resource,
            label="densities",
        )
        density_texture.copy_from_numpy(volume)

        light_dir = np.array([-0.5, 1.0, 1.0], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        light_color = np.array([30.0, 30.0, 30.0], dtype=np.float32)

        sigma_a = 0.1
        sigma_s = 10.0

        model, inv_model, view, inv_view, projection, inv_projection = setup_transforms(
            self.width, self.height, pitch, yaw, roll
        )

        for i in range(iterations):
            self.pt_kernel.dispatch(
                thread_count=[self.width, self.height, 1],
                vars={
                    "render_texture": self.render_texture,
                    "densities": density_texture,
                    "density_sampler": self.density_sampler,
                    "profile": self.profile,
                    "frame_index": i,
                    "model": model,
                    "inv_model": inv_model,
                    "view": view,
                    "inv_view": inv_view,
                    "projection": projection,
                    "inv_projection": inv_projection,
                    "light_dir": light_dir,
                    "light_color": light_color,
                    "sigma_a": sigma_a,
                    "sigma_s": sigma_s,
                },
            )

        img = self.render_texture.to_numpy()
        img = np.clip(img[..., :3] / (1.0 + img[..., :3]), 0, 1)
        img = np.dstack((img, np.ones((self.height, self.width), dtype=np.float32)))
        img[np.isnan(img)] = 0

        return img

    def isosurface(self, volume, threshold, pitch=0.0, yaw=0.0, roll=0.0):
        density_texture = self.device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=volume.shape[2],
            height=volume.shape[1],
            depth=volume.shape[0],
            usage=spy.TextureUsage.shader_resource,
            label="densities",
        )
        density_texture.copy_from_numpy(volume)

        light_dir = np.array([-0.5, 1.0, 1.0], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        light_color = np.array([30.0, 30.0, 30.0], dtype=np.float32)

        sigma_a = 0.1
        sigma_s = 10.0

        model, inv_model, view, inv_view, projection, inv_projection = setup_transforms(
            self.width, self.height, pitch, yaw, roll
        )

        self.rc_kernel.dispatch(
            thread_count=[self.width, self.height, 1],
            vars={
                "render_texture": self.render_texture,
                "densities": density_texture,
                "density_sampler": self.density_sampler,
                "profile": self.profile,
                "frame_index": 0,
                "model": model,
                "inv_model": inv_model,
                "view": view,
                "inv_view": inv_view,
                "projection": projection,
                "inv_projection": inv_projection,
                "light_dir": light_dir,
                "light_color": light_color,
                "sigma_a": sigma_a,
                "sigma_s": sigma_s,
                "threshold": threshold,
            },
        )

        img = self.render_texture.to_numpy()
        img = np.clip(img[..., :4], 0, 1)
        img[np.isnan(img)] = 0

        return img
