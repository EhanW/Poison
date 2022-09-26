from .cifar_resnet import *
from .preact_resnet import *
from .resnet import *
from .wide_resnet import *
from .vgg import *
from .policy_resnet import *

all_model_names = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'preact_resnet18', 'preact_resnet34', 'preact_resnet50',
    'preact_resnet101', 'preact_resnet152',
    'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'wide_resnet28_4', 'wide_resnet28_10', 'wide_resnet32_4', 'wide_resnet32_10',
    'mobilenet'
]

policy_model_names = ['policy_resnet18', 'policy_resnet34', 'policy_resnet50', 'policy_resnet101']
