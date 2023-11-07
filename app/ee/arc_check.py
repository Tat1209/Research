from torchvision.models import resnet18 as off_net
import utils

utils.arc_check(network=off_net(), file_name='arccheck_of.txt', verbose=1)