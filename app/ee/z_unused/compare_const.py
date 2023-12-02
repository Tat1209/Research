import utils

# from models.resnet_hase import resnet18 as hasenet
from models.resnet_hase_2d import ResNet18 as hasenet
from models.gitresnet import resnet18 as tananet
# from models.resnet_ee import resnet18 as tananet

hase = tananet(num_classes=100)
utils.arc_check(network=hase, out_file=True, file_name="tananet.txt", input_size=(100, 3, 32, 32))

hase = hasenet(num=100)
utils.arc_check(network=hase, out_file=True, file_name="hasenet.txt", input_size=(100, 3, 32, 32))
