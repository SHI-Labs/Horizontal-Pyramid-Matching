from __future__ import absolute_import

from .ResNet import *
from .ResNeXt import *
from .SEResNet import *
from .DenseNet import *
from .MuDeep import *
from .HACNN import *
from .SqueezeNet import *
from .MobileNet import *
from .ShuffleNet import *
from .Xception import *
from .InceptionV4 import *
from .NASNet import *
from .DPN import *
from .InceptionResNetV2 import *
from .HPM import * 
from .ResNetM import *
from .IDE import *
from .ResNetVideo import *

__factory = {
    # 'resnet50': resnet50,
    # 'resnet101': resnet101,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnext50': SEResNeXt50,
    'seresnext101': SEResNeXt101,
    'resnext101': ResNeXt101_32x4d,
    'resnet50m': ResNet50M,
    'densenet121': DenseNet121,
    'squeezenet': SqueezeNet,
    'mobilenet': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
    'inceptionv4': InceptionV4ReID,
    'nasnet': NASNetAMobile,
    'dpn92': DPN,
    'inceptionresnetv2': InceptionResNetV2,
    'mudeep': MuDeep,
    'hacnn': HACNN,
    'hpm': HPM,
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'resnet50rnn': ResNet50RNN,
    'resnet50ide': ResNet50IDE,
    'resnet50mide': ResNet50MIDE,
    'resnet50st':ResNet50ST,
    'resnet50sta':ResNet50STA,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)