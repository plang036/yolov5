# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.common import Conv, autopad
from utils.downloads import attempt_download

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def _scale_filters(filters, multiplier, base=8):
  """Scale the filters accordingly to (multiplier, base)."""
  round_half_up = int(int(filters) * multiplier / base + 0.5)
  result = int(round_half_up * base)
  return max(result, base)
  
def _scale(filters):
    return _scale_filters(filters, multiplier=1.0)

class Swish6(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Swish6, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return input * F.relu6(input + np.float32(3), inplace=self.inplace) * np.float32(1. / 6.)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Conv_swish6(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = Swish6() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Separable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, s=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=s, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=s, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Squeeze_and_excite(nn.Module):
    '''def _squeeze_and_excite(h, hidden_dim, activation_fn=tf.nn.relu6):
    with tf.variable_scope(None, default_name='SqueezeExcite'):
        height, width = h.shape[1], h.shape[2]
        u = slim.avg_pool2d(h, [height, width], stride=1, padding='VALID')
        u = _conv(u, hidden_dim, 1,
                normalizer_fn=None, activation_fn=activation_fn)
        u = _conv(u, h.shape[-1], 1,
                normalizer_fn=None, activation_fn=tf.nn.sigmoid)
        return u * h'''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        #kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
        self.avg_pool = nn.AvgPool2d(k, s, p)
        self.conv1 = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv2 = nn.Conv2d(c1, c1.shape[-1], k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(self.avg_pool(x))))

class Inverted_bottleneck_no_expansion(nn.Module):
    '''
    def _inverted_bottleneck_no_expansion(
    h, filters, activation_fn=tf.nn.relu6,
    kernel_size=3, strides=1, use_se=False):
    """Inverted bottleneck layer without the first 1x1 expansion convolution."""
    with tf.variable_scope(None, default_name='IBNNoExpansion'):
        # Setting filters to None will make _separable_conv a depthwise conv.
    h = _separable_conv(
        h, None, kernel_size, strides=strides, activation_fn=activation_fn)
    if use_se:
      hidden_dim = _scale_filters(h.shape[-1], 0.25)
      h = _squeeze_and_excite(h, hidden_dim, activation_fn=activation_fn)
    h = _conv(h, filters, 1, activation_fn=tf.identity)
    return h
    '''

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, use_se=True, bias=False):
        super().__init__()
        filter = self.scale(c2)
        self.separable_conv = Separable_Conv2d(c1, filter, k, bias)
        self.squeeze_and_excite = Squeeze_and_excite(c1, self.scale_filters(c1.shape[-1], 0.25))
        self.conv = nn.Conv2d(c1, filter, 1, s, autopad(k, p), groups=g, bias=False)
        self.use_se = use_se
        self.act = Swish6()

    def forward(self, x):
        _x = self.separable_conv(x)
        if self.use_se:
            _x = self.squeeze_and_excite(_x)
        return self.act(self.conv(_x))

    def scale_filters(self, filters, multiplier, base=8):
        """Scale the filters accordingly to (multiplier, base)."""
        round_half_up = int(int(filters) * multiplier / base + 0.5)
        result = int(round_half_up * base)
        return max(result, base)
        
    def scale(self, filters):
        return self.scale_filters(filters, multiplier=1.0)


class Inverted_bottleneck(nn.Module):
    '''
    def _inverted_bottleneck(
        h, filters, activation_fn=tf.nn.relu6,
        kernel_size=3, expansion=8, strides=1, use_se=False, residual=True):
    """Inverted bottleneck layer."""
    with tf.variable_scope(None, default_name='IBN'):
        shortcut = h
        expanded_filters = int(h.shape[-1]) * expansion
        if expansion <= 1:
        raise ValueError('Expansion factor must be greater than 1.')
        h = _conv(h, expanded_filters, 1, activation_fn=activation_fn)
        # Setting filters to None will make _separable_conv a depthwise conv.
        h = _separable_conv(h, None, kernel_size, strides=strides,
                            activation_fn=activation_fn)
        if use_se:
            hidden_dim = _scale_filters(expanded_filters, 0.25)
            h = _squeeze_and_excite(h, hidden_dim, activation_fn=activation_fn)
        h = _conv(h, filters, 1, activation_fn=tf.identity)
        if residual:
        h = h + shortcut
        return h
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, expansion=8, residual=True, use_se=False, g=1):
        super().__init__()
        self.use_se = use_se
        self.residual = residual
        filter = self.scale(c2)
        self.expanded_filters = int(c1.shape[-1]) * expansion
        self.squeeze_and_excite = Squeeze_and_excite(c1, self.scale_filters(self.expanded_filters, 0.25))
        self.conv1 = nn.Conv2d(c1, self.expanded_filters, 1, 1, autopad(k, p), groups=g, bias=False)
        self.separable_conv = Separable_Conv2d(c1, filter, k, s)
        self.conv2 = nn.Conv2d(c1, filter, 1)
        self.act = Swish6()


    def forward(self, x):
        _x = self.separable_conv(self.conv1(x))
        if self.use_se:
            _x = self.squeeze_and_excite(_x)
        _x = self.conv2(_x)
        if self.residual:
            _x = _x + x
        return self.act(_x)

    def scale_filters(self, filters, multiplier, base=8):
        """Scale the filters accordingly to (multiplier, base)."""
        round_half_up = int(int(filters) * multiplier / base + 0.5)
        result = int(round_half_up * base)
        return max(result, base)
        
    def scale(self, filters):
        return self.scale_filters(filters, multiplier=1.0)


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse


    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
