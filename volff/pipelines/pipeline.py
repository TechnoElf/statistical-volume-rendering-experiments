import importlib.resources as resources
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import slangpy as spy

import volff


class PipelineCtx:
    def __init__(self):
        self.kernel_path_ctx_manager = resources.path(volff, "kernels")

    def __enter__(self) -> Self:
        self.kernel_path_ctx = self.kernel_path_ctx_manager.__enter__()
        self.device = spy.create_device(
            type=spy.DeviceType.vulkan,
            include_paths=[self.kernel_path_ctx],
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.kernel_path_ctx_manager.__exit__(exc_type, exc_value, traceback)


class Pipeline(ABC):
    def __init__(self, ctx: None | PipelineCtx = None):
        self.ctx_manager = None
        self.ctx = ctx

    def __enter__(self) -> Self:
        if self.ctx is None:
            self.ctx_manager = PipelineCtx()
            self.ctx = self.ctx_manager.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.ctx_manager is not None:
            self.ctx_manager.__exit__(exc_type, exc_value, traceback)

    @abstractmethod
    def prepare(self, volume: np.ndarray, width: int = 1280, height: int = 720):
        pass

    @abstractmethod
    def render(self, params: dict) -> np.ndarray:
        pass
