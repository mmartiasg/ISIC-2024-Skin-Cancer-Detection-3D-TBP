import torch
import torchvision
from prototypes.utility.data import ProjectConfiguration

config = ProjectConfiguration("config.json")


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
    def __init__(self, num_classes, in_features=768, out_features=512):
        super(ClassificationHeadVit_16, self).__init__()
        self.num_classes = num_classes
        self.ffc1 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc2 = torch.nn.Linear(in_features=out_features, out_features=self.num_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
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
        self.model.fc = ClassificationHeadDropout(self.n_classes, dropout_rate=0.1)
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
        self.model.fc = ClassificationHeadDropout(self.n_classes, dropout_rate=0.2)
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
        # self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.heads = ClassificationHeadDropoutVit_16(self.n_classes, dropout_rate=0.10)

    def forward(self, x):
        return self.model(x)


class VitPrototype1(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitPrototype1, self).__init__()
        # self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.model.heads = ClassificationHeadVit_16(self.n_classes)

    def forward(self, x):
        return self.model(x)


class VitMetadata(torch.nn.Module):
    def __init__(self, n_classes):
        super(VitMetadata, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        self.metadata_encoding_layer = torch.nn.Linear(in_features=16, out_features=32)
        #remove bias
        self.mixing_layer = torch.nn.Linear(in_features=800, out_features=768, bias=False)
        self.activation_layer = torch.nn.ReLU(inplace=True)
        self.classification_head = ClassificationHeadVit_16(self.n_classes, in_features=768, out_features=512)
        self.batch_norm = torch.nn.BatchNorm1d(num_features=768)
        self.model.heads = torch.nn.Identity()

    def forward(self, x):
        image_x = x[0]
        metadata_x = x[1]

        image_encoding = self.model(image_x)
        # metadata_x = self.batch_norm(metadata_x)
        metadata_encoding = self.metadata_encoding_layer(metadata_x)
        concat = torch.cat((image_encoding, metadata_encoding), 1)
        mix = self.mixing_layer(concat)
        mix = self.batch_norm(mix)
        mix_activation = self.activation_layer(mix)

        return self.classification_head(mix_activation)


class Vit_b_16_MHA(torch.nn.Module):
    def __init__(self, n_classes):
        super(Vit_b_16_MHA, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(num_classes=n_classes)
        self.proy_metadata = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=64),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Linear(in_features=64, out_features=64))
        self.n_classes = n_classes
        # The data will be recenter by the batch norm thus we can avoid the bias since it will be removed
        self.metadata_encoding_layer = torch.nn.Linear(in_features=32, out_features=64, bias=False)
        self.classification_head = ClassificationHeadVit_16(self.n_classes, in_features=768, out_features=512)
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=64)
        self.batch_norm_metadata_proy = torch.nn.BatchNorm1d(num_features=64)
        self.self_attention_q = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.decoder_mha = torch.nn.MultiheadAttention(embed_dim=64, kdim=768, vdim=768, num_heads=8)
        self.model.heads = torch.nn.Identity()
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=64)
        self.batch_norm3 = torch.nn.BatchNorm1d(num_features=768)
        self.mixing_layer = torch.nn.Linear(in_features=832, out_features=768, bias=False)

    def forward(self, x):
        image_x = x[0]
        metadata_x = x[1]

        image_encoding = self.model(image_x)

        metadata_x = self.metadata_encoding_layer(metadata_x)
        metadata_encoding, _ = self.self_attention_q(query=metadata_x, key=metadata_x, value=metadata_x, average_attn_weights=True)
        residual_metadata_encoding = self.batch_norm1(metadata_x + metadata_encoding)
        proy_metadata_out = self.proy_metadata(residual_metadata_encoding)
        proy_metadata_out_n = self.batch_norm_metadata_proy(proy_metadata_out)

        q_v_attention, _ = self.decoder_mha(query=proy_metadata_out_n, key=image_encoding, value=image_encoding, average_attn_weights=True)
        residual_q_v = self.batch_norm2(q_v_attention + residual_metadata_encoding)

        mix = self.mixing_layer(torch.cat((residual_q_v, image_encoding), dim=1))
        mix = self.batch_norm3(mix)
        return self.classification_head(mix)


class Vit_b_16_MHA_V2(torch.nn.Module):
    def __init__(self, n_classes):
        super(Vit_b_16_MHA_V2, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(num_classes=n_classes)
        self.n_classes = n_classes
        self.set_up()

    def set_up(self):
        # The data will be recenter by the batch norm thus we can avoid the bias since it will be removed
        self.metadata_encoding_layer = torch.nn.Linear(in_features=24, out_features=64, bias=False)
        self.self_attention_q = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8)

        self.proy_metadata = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=64),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Linear(in_features=64, out_features=64))

        self.batch_norm_metadata_1 = torch.nn.BatchNorm1d(num_features=64)
        self.batch_norm_metadata_2 = torch.nn.BatchNorm1d(num_features=64)

        self.decoder_mha = torch.nn.MultiheadAttention(embed_dim=64, kdim=768, vdim=768, num_heads=8)

        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=64)

        self.proy_metadata_image_combined = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=64),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Linear(in_features=64, out_features=64))

        # self.batch_norm3 = torch.nn.BatchNorm1d(num_features=768)

        self.model.heads = torch.nn.Identity()
        # self.mixing_layer = torch.nn.Linear(in_features=832, out_features=768, bias=False)
        # self.classification_head = ClassificationHeadVit_16(self.n_classes, in_features=768, out_features=512)
        self.classification_head = ClassificationHeadVit_16(self.n_classes, in_features=64, out_features=32)

    def forward(self, x):
        image_x = x[0]
        metadata_x = x[1]

        metadata_x = self.metadata_encoding_layer(metadata_x)
        metadata_encoding, _ = self.self_attention_q(query=metadata_x, key=metadata_x, value=metadata_x,
                                                     average_attn_weights=True)
        proy_metadata_in = self.batch_norm_metadata_1(metadata_x + metadata_encoding)
        proy_metadata_out = self.proy_metadata(proy_metadata_in)
        residual_metadata_proy = self.batch_norm_metadata_2(proy_metadata_in + proy_metadata_out)

        image_encoding = self.model(image_x)
        q_v_attention, _ = self.decoder_mha(query=residual_metadata_proy, key=image_encoding, value=image_encoding,
                                            average_attn_weights=True)
        proy_metadata_image_in = self.batch_norm2(q_v_attention + residual_metadata_proy)
        proy_metadata_image_out = self.proy_metadata_image_combined(proy_metadata_image_in)

        # mix = self.mixing_layer(torch.cat((proy_metadata_image_out, image_encoding), dim=1))
        # mix = self.batch_norm3(mix)
        # return self.classification_head(mix)

        return self.classification_head(proy_metadata_image_out)

class Vit16(torch.nn.Module):
    def __init__(self, n_classes):
        super(Vit16, self).__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.model = torchvision.models.vit_b_16(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)


class MaxVit(torch.nn.Module):
    def __init__(self, n_classes):
        super(MaxVit, self).__init__()
        self.weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1
        self.model = torchvision.models.maxvit_t(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)


class SwingB(torch.nn.Module):
    def __init__(self, n_classes):
        super(SwingB, self).__init__()
        self.weights = torchvision.models.Swin_B_Weights.IMAGENET1K_V1
        self.model = torchvision.models.swin_b(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)


class SwingV2B(torch.nn.Module):
    def __init__(self, n_classes):
        super(SwingV2B, self).__init__()
        self.weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1
        self.model = torchvision.models.swin_v2_b(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)


class ResNex10164x4d(torch.nn.Module):
    def __init__(self, n_classes):
        super(ResNex10164x4d, self).__init__()
        self.weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnext101_64x4d(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)


class WideResNet101(torch.nn.Module):
    def __init__(self, n_classes):
        super(WideResNet101, self).__init__()
        self.weights = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        self.model = torchvision.models.wide_resnet101_2(num_classes=n_classes)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x)
        return self.out(logit)