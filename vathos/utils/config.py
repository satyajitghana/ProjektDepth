
import yaml

from typing import Any, List, Tuple, Dict
from types import ModuleType

import torch
import torch.nn as nn

from .logger import setup_logger

logger = setup_logger(__name__)


def load_config(filename: str) -> dict:
    r"""Load a configuration file as YAML and returns a dict

    Args:
        filename (str): location of the file

    Returns:
        (Dict) of the config
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    return config


def setup_device(model: nn.Module, target_device: int) -> Tuple[torch.device, List[int]]:
    r"""sets up the device for the model

    Args:
        model (nn.Module): the model
        target_device (int): index of the target device

    Returns:
        Tuple[torch.device, List[int]]
    """
    available_devices: List = list(range(torch.cuda.device_count()))
    logger.info(
        f'Using device {target_device} of available devices {available_devices}')

    device = torch.device(f'cuda:{target_device}')
    model = model.to(device)

    return model, device


def setup_param_groups(model: nn.Module, config: Dict) -> List:
    return [{'params': model.parameters(), **config}]


def get_instance(module: ModuleType, name: str, config: Dict, *args: Any) -> Any:
    r"""creates an instance from a constructor name and module name

    Args:
        module (ModuleType): the module which contains the class
        name (str): name of the class
        config (Dict): configuration of experiment
        args (Any): any arguments that needs to be passed to the class
    """
    ctor_name = config[name]['type']
    logger.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


def get_instance_v2(module, ctor_name, *args, **kwargs):
    r"""creates an instance from a constructor name and module name

    Args:
        module: the module which contains the ctor_name
        ctor_name: name of the constructor
        args(Optional): positional arguments that needs to be passed to ctor
        kwargs(Optional): keywords arguments that needs to be passed to ctor
    """
    logger.info(f'Building {module.__name__}.{ctor_name}')

    return getattr(module, ctor_name)(*args, **kwargs)
