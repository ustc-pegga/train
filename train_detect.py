import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# 选择新的骨干网络，例如 MobileNetV3
new_backbone = torchvision.models.mobilenet_v2(pretrained=True)

# 加载 SSD Lite 模型
ssd_lite_model = ssdlite320_mobilenet_v3_large(pretrained=True)

# 替换 SSD Lite 中的骨干网络
ssd_lite_model.backbone = new_backbone.features

# 可以选择冻结新骨干网络的参数，以防止在训练时更新
for param in ssd_lite_model.backbone.parameters():
    param.requires_grad = False

# 可选：修改 SSD Lite 输出的类别数（默认为 21，包括背景类）
# 例如，如果您的任务有 10 个类别，可以进行如下修改：
ssd_lite_model.box_predictor = nn.Conv2d(1280, 4 * (4 + 1 + 10), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
print(ssd_lite_model)
# 现在，您可以使用 ssd_lite_model 进行目标检测训练和推理