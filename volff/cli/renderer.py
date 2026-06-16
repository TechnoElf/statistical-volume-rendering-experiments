from enum import Enum

import numpy as np

from volff.pipelines.levoy import LevoyPipeline
from volff.pipelines.pathtrace import PathTracePipeline
from volff.pipelines.relight import RelightPipeline


class RenderKind(str, Enum):
    pathtrace = "pathtrace"
    levoy = "levoy"
    relight = "relight"


def render(kind: RenderKind, volume: np.ndarray, params: dict) -> np.ndarray:
    if kind == RenderKind.pathtrace:
        p = PathTracePipeline()
    elif kind == RenderKind.levoy:
        p = LevoyPipeline()
    elif kind == RenderKind.relight:
        p = RelightPipeline()

    with p:
        p.prepare(volume)
        return p.render(params)
