import torch

from model.ood_detector.aps import APS
from model.ood_detector.msp import MSP
from model.ood_detector.nnguide import NNGuide
from model.ood_detector.energy import Energy


class OODDetector(object):
    OOD_ALGORITHM = {
        "aps": APS,
        "msp": MSP,
        "nnguide": NNGuide,
        "energy": Energy
    }

    def __new__(cls, algorithm, *args, **kwargs):
        dataset_factory = cls.OOD_ALGORITHM.get(algorithm)
        assert dataset_factory, f"{algorithm} is not recognized as valid task"
        return dataset_factory(*args, **kwargs)
    