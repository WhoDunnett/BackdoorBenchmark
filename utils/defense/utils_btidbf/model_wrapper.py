import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActResNetWrapper(nn.Module):
    def __init__(self, model):
        super(PreActResNetWrapper, self).__init__()
        self.model = model
        # For torchvision model, the typical structure is:
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.feature_extractor = nn.Sequential(
            self.model.conv1,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )

        self.avgpool = self.model.avgpool
        self.linear = self.model.linear

    def forward(self, x):
        features = self.from_input_to_features(x)
        return self.from_features_to_output(features)

    def from_input_to_features(self, x):
        # Pass input through the CNN backbone
        return self.feature_extractor(x)

    def from_features_to_output(self, features):
        # Finish with pooling and the classifier (dense layers)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        return self.linear(x)
    
class EfficientNetWrapper(nn.Module):
    def __init__(self, model):
        super(EfficientNetWrapper, self).__init__()
        self.model = model
        # Correct the reference: use self.model.features instead of self.features.
        self.feature_extractor = self.model.features  
        self.avgpool = self.model.avgpool
        self.linear = self.model.classifier  # torchvision's EfficientNet uses "classifier"
    
    def forward(self, x):
        features = self.from_input_to_features(x)
        return self.from_features_to_output(features)
    
    def from_input_to_features(self, x):
        return self.feature_extractor(x)
    
    def from_features_to_output(self, features):
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        return self.linear(x)

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, model):
        super(MobileNetV3Wrapper, self).__init__()
        self.model = model
        # Correct the reference: use self.model.features instead of self.features.
        self.feature_extractor = self.model.features  
        self.avgpool = self.model.avgpool
        self.linear = self.model.classifier  # MobileNetV3 typically uses "classifier"
    
    def forward(self, x):
        features = self.from_input_to_features(x)
        return self.from_features_to_output(features)
    
    def from_input_to_features(self, x):
        return self.feature_extractor(x)
    
    def from_features_to_output(self, features):
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        return self.linear(x)
    
class VGGWrapper(nn.Module):
    def __init__(self, model):
        super(VGGWrapper, self).__init__()
        self.model = model
        # Correct the reference: use self.model.features instead of self.features.
        self.feature_extractor = self.model.features  
        self.avgpool = self.model.avgpool
        self.linear = self.model.classifier  # MobileNetV3 typically uses "classifier"
    
    def forward(self, x):
        features = self.from_input_to_features(x)
        return self.from_features_to_output(features)
    
    def from_input_to_features(self, x):
        return self.feature_extractor(x)
    
    def from_features_to_output(self, features):
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        return self.linear(x)