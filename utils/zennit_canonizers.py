import torch
from zennit.canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from zennit.layer import Sum


class YoloCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        # print(module.__class__.__name__)
        if module.__class__.__name__ == "Bottleneck":
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if module.__class__.__name__ == "RepVGGBlock":
            attributes = {
                'forward': cls.forward_.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        if not self.add:
            return self.cv2(self.cv1(x))
        else:
            x = torch.stack([x, self.cv2(self.cv1(x))], dim=-1)
            x = self.canonizer_sum(x)
            return x

    @staticmethod
    def forward_(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        x = torch.stack([self.rbr_dense(inputs), self.rbr_1x1(inputs), id_out]
                        if self.rbr_identity is not None else [self.rbr_dense(inputs), self.rbr_1x1(inputs)], dim=-1)
        x = self.canonizer_sum(x)
        return self.nonlinearity(self.se(x))


class YoloV5V6Canonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            YoloCanonizer(),
        ))


class DeepLabV3PlusBottleneckCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        # print(module.__class__.__name__)
        if module.__class__.__name__ == "Bottleneck":
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)
        out = self.relu(out)

        return out


class DeepLabV3PlusCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            DeepLabV3PlusBottleneckCanonizer()
        ))
