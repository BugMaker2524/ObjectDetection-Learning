import torch.nn as nn
import torch


def BasicConv2d(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class InceptionV1Module(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce,
                 out_channels3, out_channels4):
        super().__init__()
        # 线路1，单个1×1卷积层
        self.branch1_conv = BasicConv2d(in_channels, out_channels1, kernel_size=1)
        # 线路2，1×1卷积层后接3×3卷积层
        self.branch2_conv1 = BasicConv2d(in_channels, out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = BasicConv2d(out_channels2reduce, out_channels2, kernel_size=3)
        # 线路3，1×1卷积层后接5×5卷积层
        self.branch3_conv1 = BasicConv2d(in_channels, out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = BasicConv2d(out_channels3reduce, out_channels3, kernel_size=5)
        # 线路4，3×3最大池化层后接1×1卷积层
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = BasicConv2d(in_channels, out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception3 = nn.Sequential(
            InceptionV1Module(192, 64, 96, 128, 16, 32, 32),
            InceptionV1Module(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception4 = nn.Sequential(
            InceptionV1Module(480, 192, 96, 208, 16, 48, 64),
            InceptionV1Module(512, 160, 112, 224, 24, 64, 64),
            InceptionV1Module(512, 128, 128, 256, 24, 64, 64),
            InceptionV1Module(512, 112, 144, 288, 32, 64, 64),
            InceptionV1Module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception5 = nn.Sequential(
            InceptionV1Module(832, 256, 160, 320, 32, 128, 128),
            InceptionV1Module(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


net = GoogLeNet(1000)
X = torch.rand(1, 3, 224, 224)
for block in net.children():
    X = block(X)
    print('output shape: ', X.shape)
