import torch
from zennit.composites import SpecialFirstLayerMapComposite, LAYER_MAP_BASE, LayerMapComposite
from zennit.layer import Sum
from zennit.rules import ZPlus, Epsilon, Flat, Pass, Norm, ReLUGuidedBackprop
from zennit.types import Convolution, Linear, Activation, AvgPool, BatchNorm

from models.yolov5 import Sigmoid_
from utils.zennit_rules import GammaResNet


class EpsilonPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
            (BatchNorm, Epsilon()),
            (Sigmoid_, Pass())
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class EpsilonGammaFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, GammaResNet()),
            (torch.nn.Linear, Epsilon()),
            (BatchNorm, Epsilon()),
            (Sigmoid_, Pass())
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class EpsilonFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Epsilon()),
            (torch.nn.Linear, Epsilon()),
            (BatchNorm, Epsilon()),
            (Sigmoid_, Pass())
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class EpsilonPlus(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
            (BatchNorm, Epsilon()),
            (Sigmoid_, Pass())
        ]
        first_map = [
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class AllFlatComposite(LayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer
    '''

    def __init__(self, canonizers=None):
        layer_map = [
            (Linear, Flat()),
            (AvgPool, Flat()),
            (torch.nn.modules.pooling.MaxPool2d, Flat()),
            (Activation, Pass()),
            (Sum, Flat()),
            (Sigmoid_, Pass()),
            (BatchNorm, Pass()),
        ]

        super().__init__(layer_map, canonizers=canonizers)

class GradientComposite(LayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer
    '''

    def __init__(self, canonizers=None):
        layer_map = [
            (Sigmoid_, Pass()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


class GuidedBackpropComposite(LayerMapComposite):
    '''An explicit composite modifying the gradients of all ReLUs according to GuidedBackprop
    :cite:p:`springenberg2015striving`.
    '''
    def __init__(self, canonizers=None):
        layer_map = [
            (torch.nn.ReLU, ReLUGuidedBackprop()),
            (Sigmoid_, Pass()),
        ]
        super().__init__(layer_map, canonizers=canonizers)