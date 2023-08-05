import torch
import torch.nn as nn

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, img_shape, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super(SiameseNetwork, self).__init__()
        
        self.img_channels = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[2]

        if backbone in models.__dict__:
            # Create a backbone network from the pretrained models provided in torchvision.models 
            self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
        else: 
            print(f"No model named {backbone} exists in torchvision.models.")
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=11,stride=4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2),
                
                nn.Conv2d(96, 256, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256, 384, kernel_size=3,stride=1),
                nn.ReLU(inplace=True)
            )

        # Get the number of features that are outputted by the last layer of backbone network.
        # out_features = list(self.backbone.modules())[-1].out_features
        out_features = self._get_out_features()
        
        
        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Linear(out_features, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,2)
        )
        
    def _get_out_features(self):
        # Calculate the output shape of the last convolutional layer
        dummy_input = torch.zeros(1, self.img_channels, self.img_height, self.img_width)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
            
        _, channels, height, width = dummy_output.shape

        # Calculate the input size for the fully connected layers
        return channels * height * width

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
        image_embed1 = image_embed1.view(image_embed1.size()[0], -1)
        image_embed2 = image_embed2.view(image_embed2.size()[0], -1)
        image_embed1 = self.cls_head(image_embed1)
        image_embed2 = self.cls_head(image_embed2)
        
        return image_embed1, image_embed2
        
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
        # combined_features = image_embed1 * image_embed2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        # output = self.cls_head(combined_features)
        # return output