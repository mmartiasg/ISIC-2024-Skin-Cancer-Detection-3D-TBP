import torch
import torchvision


class ClassificationHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.ffc1 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc2 = torch.nn.Linear(in_features=1024, out_features=self.num_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ffc1(x)
        x = self.relu(x)
        x = self.ffc2(x)
        return self.out(x)


class ClassificationHeadDropout(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(ClassificationHeadDropout, self).__init__()
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.ffc1 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc2 = torch.nn.Linear(in_features=1024, out_features=self.num_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.ffc1(x)
        x = self.relu(x)
        x = self.ffc2(x)
        return self.out(x)


class ClassificationHeadDropoutVit_16(torch.nn.Module):
    def __init__(self, num_classes, in_features=768, out_features=512, dropout_rate=0.4):
        super(ClassificationHeadDropoutVit_16, self).__init__()
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.ffc1 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc2 = torch.nn.Linear(in_features=out_features, out_features=self.num_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.ffc1(x)
        x = self.relu(x)
        x = self.ffc2(x)
        return self.out(x)

class ClassificationHeadVit_16(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(ClassificationHeadDropoutVit_16, self).__init__()
        self.num_classes = num_classes
        self.ffc1 = torch.nn.Linear(in_features=768, out_features=512)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc2 = torch.nn.Linear(in_features=512, out_features=self.num_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.ffc1(x)
        x = self.relu(x)
        x = self.ffc2(x)
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
        self.model.fc = ClassificationHeadDropout(self.n_classes, dropout_rate=0.7)
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


class Resnet50Prototype2Dropout(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype2Dropout, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHeadDropout(self.n_classes, dropout_rate=0.6)
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


class Resnet50Prototype3Dropout(torch.nn.Module):
    def __init__(self, n_classes):
        super(Resnet50Prototype3Dropout, self).__init__()
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.fc = ClassificationHeadDropout(self.n_classes, dropout_rate=0.6)
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class VitPrototype1Dropout(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitPrototype1Dropout, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.heads = ClassificationHeadDropoutVit_16(self.n_classes, dropout_rate=0.6)

    def forward(self, x):
        return self.model(x)


class VitPrototype2Dropout(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitPrototype2Dropout, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.metadata_encoding_layer = torch.nn.Linear(in_features=32, out_features=32)
        self.mixing_layer = torch.nn.Linear(in_features=800, out_features=768)
        self.activation_layer = torch.nn.ReLU(inplace=True)
        self.classification_head = ClassificationHeadDropoutVit_16(self.n_classes, dropout_rate=0.6)
        self.batch_norm = torch.nn.BatchNorm1d(num_features=32)
        self.model.heads = torch.nn.Identity()

    def forward(self, x):
        image_x = x[0]
        metadata_x = x[1]

        image_encoding = self.model(image_x)
        metadata_x = self.batch_norm(metadata_x)
        metadata_encoding = self.metadata_encoding_layer(metadata_x)
        concat = torch.cat((image_encoding, metadata_encoding), 1)
        mix = self.mixing_layer(concat)
        mix_activation = self.activation_layer(mix)

        return self.classification_head(mix_activation)


class VitPrototype3MHA(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitPrototype3MHA, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        # self.metadata_encoding_layer = torch.nn.Linear(in_features=32, out_features=32)
        self.classification_head = ClassificationHeadDropoutVit_16(self.n_classes, in_features=768, out_features=512, dropout_rate=0.6)
        self.batch_norm = torch.nn.BatchNorm1d(num_features=32)
        self.self_attention_q = torch.nn.MultiheadAttention(embed_dim=32, num_heads=8, dropout=0.1)
        self.decoder_mha = torch.nn.MultiheadAttention(embed_dim=32, kdim=768, vdim=768, num_heads=8)
        self.model.heads = torch.nn.Identity()
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=32)
        self.mixing_layer = torch.nn.Linear(in_features=800, out_features=768)

    def forward(self, x):
        image_x = x[0]
        metadata_x = x[1]

        image_encoding = self.model(image_x)
        metadata_x = self.batch_norm(metadata_x)
        metadata_encoding, _ = self.self_attention_q(query=metadata_x, key=metadata_x, value=metadata_x, average_attn_weights=False)
        residual_metadata_encoding = self.batch_norm2(metadata_x + metadata_encoding)
        attention, _ = self.decoder_mha(query=residual_metadata_encoding, key=image_encoding, value=image_encoding, average_attn_weights=False)
        mix = self.mixing_layer(torch.cat((attention, image_encoding), dim=1))
        return self.classification_head(mix)


class VitPrototype1(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitPrototype1, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.heads = ClassificationHeadVit_16(self.n_classes, dropout_rate=0.6)

    def forward(self, x):
        return self.model(x)
