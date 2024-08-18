import torch
import torchvision


class ClassificationHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(in_features=2048, out_features=self.num_classes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return self.out(x)


class ClassificationHeadDropout(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHeadDropout, self).__init__()
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(p=0.4)
        self.ffc1 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.ffc1(x)
        x = self.relu(x)
        return self.out(x)


class Resnet50Prototype1(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype1, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHead(self.n_classes)
        for param in self.model.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class Resnet50Prototype1Dropout(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype1Dropout, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHeadDropout(self.n_classes)
        for param in self.model.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class Resnet50Prototype2(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype2, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHead(self.n_classes)
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class Resnet50Prototype3(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype3, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHead(self.n_classes)
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
