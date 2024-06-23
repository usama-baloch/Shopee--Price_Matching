import torch
import math
import torch.nn.functional as F

class ArcMarginProduct(torch.nn.Module):

    def __init__(self, in_features, out_features, scale=30.0,
                 margin=0.5, easy_margin=False, ls_eps = 0.0):
        
        super(ArcMarginProduct, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))

        torch.nn.init.xavier_uniform(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):

        # calculate the cosine
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # calculate the sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # calculate the phi
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phir = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device = 'cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output, torch.nn.CrossEntropyLoss()(output, label)
    