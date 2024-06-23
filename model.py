import torch
import torch.nn as nn
import config
from arcprod import ArcMarginProduct

class ShopeeModel(nn.Module):

    def __init__(self, n_classes = config.classes, model_name = config.model_name,
                fc_dim = config.FC_DIM, scale = config.Scale, margin = config.Margin,
                use_fc = True, pretrained = True):
        
        super(ShopeeModel, self).__init__()
        print(f'Building up the Model {model_name}')

        self.backbone = timm.create_model(model_name, pretrained = pretrained)

        if 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()
        
        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()
        
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if self.use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm2d(fc_dim)
            self._init_params()
            final_in_features = fc_dim
        
        self.final = ArcMarginProduct(final_in_features, 
                                      n_classes,
                                      scale = scale,
                                      margin = margin,
                                      easy_margin = False,
                                      ls_eps = 0.0)
        
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias,0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feature(image)
        logits = self.final(feature, label)

        return logits
    
    def extract_feature(self, x):
        batch_size = x.shape[0]
        out = self.backbone(x)
        out = self.pooling(out).view(batch_size, -1)

        if self.use_fc:
            out = self.dropout(out)
            out = self.fc(out)
            out = self.bn(out)

        return out