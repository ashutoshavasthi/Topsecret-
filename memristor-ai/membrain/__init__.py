"""MemBrain: neural networks that compute and learn inside simulated memristor crossbars."""
from .crossbar import IDEAL_PERIPHERY, Crossbar, Periphery
from .device import PRESETS, DeviceArray, DeviceSpec, get_device
from .network import MLP, TrainConfig, analog_mlp, float_mlp, to_analog, train

__all__ = ["Crossbar", "Periphery", "IDEAL_PERIPHERY", "DeviceSpec", "DeviceArray", "PRESETS", "get_device",
           "MLP", "TrainConfig", "train", "float_mlp", "analog_mlp", "to_analog"]
__version__ = "0.1.0"
