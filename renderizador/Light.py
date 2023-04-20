import numpy as np
from typing import List

class Light():
    def __init__(self, ambientIntensity: float, color: List[float], intensity: float, direction: List[float]) -> None:
        self.ambientIntensity = ambientIntensity
        self.color = color
        self.intensity = intensity
        self.direction = direction

    def __str__(self) -> str:
        return "Light:\nColor:{}\tIntensity:{}\tDirection:{}".format(self.color, self.intensity, self.direction)

    def __repr__(self) -> str:
        return str(self)

class DirectionalLight(Light):
    pass