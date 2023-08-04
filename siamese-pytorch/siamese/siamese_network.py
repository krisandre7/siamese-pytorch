import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features
        
        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.ReLU(inplace=True)
        )

    def forward(self, image1, image2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    image1 (torch.Tensor)
                    image2 (torch.Tensor)


            Returns:
                    image_embed1 (torch.Tensor), vector representation of image1
                    image_embed2 (torch.Tensor), vector representation of image2
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        image_embed1 = self.backbone(image1)
        image_embed2 = self.backbone(image2)
        
        # Pass both images through a perceptron to generate embeds
        image_embed1 = self.cls_head(image_embed1)
        image_embed2 = self.cls_head(image_embed2)
        
        similarity = F.pairwise_distance(image_embed1, image_embed2, p=2, keepdim=True)
        return similarity
        
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
        # combined_features = image_embed1 * image_embed2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        # output = self.cls_head(combined_features)
        # return output