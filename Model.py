from model_utils import *
import pdb
import os
import torch.nn.functional as F

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            channel: (todo): write your description
            reduction: (todo): write your description
        """
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        """
        Forward computation of the image.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y

# Grad Reversal
class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        """
        Initialize the dsbd

        Args:
            self: (todo): write your description
            lambd: (float): write your description
        """
        self.lambd = lambd

    def forward(self, x):
        """
        Return the next item from x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x.view_as(x)

    def backward(self, grad_output):
        """
        Perform backward backward pass.

        Args:
            self: (todo): write your description
            grad_output: (bool): write your description
        """
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    """
    Reverse gradient.

    Args:
        x: (int): write your description
        lambd: (todo): write your description
    """
    return GradReverse(lambd)(x)

# Generator
class Pointnet_g(nn.Module):
    def __init__(self):
        """
        Initialize layer

        Args:
            self: (todo): write your description
        """
        super(Pointnet_g, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node = False):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            node: (todo): write your description
        """
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        x, node_fea, node_off = self.conv3(x, x_loc)  # x = [B, dim, num_node, 1]/[64, 64, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv5(x)

        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)
  
        x = self.bn1(x)

        if node == True:
            return x, node_fea, node_off
        else:
            return x, node_fea

# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, num_class=10):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            num_class: (int): write your description
        """
        super(Pointnet_c, self).__init__()
        self.fc = nn.Linear(1024, num_class)
        
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.fc(x)
        return x
        
class Net_MDA(nn.Module):
    def __init__(self, model_name='Pointnet'):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            model_name: (str): write your description
        """
        super(Net_MDA, self).__init__()
        if model_name == 'Pointnet':
            self.g = Pointnet_g() 
            self.attention_s = CALayer(64*64)
            self.attention_t = CALayer(64*64)
            self.c1 = Pointnet_c()  
            self.c2 = Pointnet_c() 
            
    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False, node_adaptation_t=False):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            constant: (todo): write your description
            adaptation: (todo): write your description
            node_vis: (todo): write your description
            mid_feat: (todo): write your description
            node_adaptation_s: (todo): write your description
            node_adaptation_t: (todo): write your description
        """
        x, feat_ori, node_idx = self.g(x, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis ==True:
            return node_idx

        # collect mid-level feat
        if mid_feat == True:
            return x, feat_ori
        
        if node_adaptation_s == True:
            # source domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s
        elif node_adaptation_t == True:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_t

        if adaptation == True:
            x = grad_reverse(x, constant)

        y1 = self.c1(x)
        y2 = self.c2(x)
        return y1, y2
