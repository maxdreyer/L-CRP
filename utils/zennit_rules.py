import torch
from zennit.core import BasicHook, stabilize


class GammaResNet(BasicHook):
    '''LRP Gamma rule :cite:p:`montavon2019layer`.

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    '''
    def __init__(self, gamma=0.25):
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[lambda param, _: param + gamma * param.clamp(min=0),
                             lambda param, name: param + gamma * param.clamp(max=0) if name != 'bias' else torch.zeros_like(param),
                             lambda param, _: param + gamma * param.clamp(max=0),
                             lambda param, name: param + gamma * param.clamp(min=0) if name != 'bias' else torch.zeros_like(param),
                             lambda param, _: param,
                             ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * out_grad / stabilize(denom)
                    for output, denom in (
                        [(outputs[4] > 0., sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0., sum(outputs[2:4]))] * 2
                    )
                ] + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(input * gradient for input, gradient in zip(inputs[:4], gradients[:4]))
            ),
        )