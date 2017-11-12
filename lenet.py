import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Input - 1x28x28
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, input):
        output = self.c1(input)
        output = F.tanh(output)
        output = self.s2(output)
        output = self.c3(output)
        output = F.tanh(output)
        output = self.s4(output)
        output = self.c5(output)
        output = output.view(-1, 120)
        output = self.f6(output)
        output = F.tanh(output)
        output = self.f7(output)
        output = F.softmax(output)
        return output


