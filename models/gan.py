import torch
import torch.nn as nn
import torch.nn.functional as F
from models.inn import SequenceINN, AllInOneBlock, PermuteRandom

# ========================================= Weight Initialization =========================================

def weights_init(m):
    """
    Initializes weights of linear layers with normal distribution and biases with zeros.

    Args:
    - m (Module): PyTorch module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# ================================== Invertible Generative Adverserial Network ==================================

class FeatureEmbeddingNet(nn.Module):
    """
    A network for embedding input features into a latent space.
    
    Args:
        opt: Configuration options including sizes for input, hidden, and output layers.
    """
    def __init__(self, opt):
        super(FeatureEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(opt.image_feature_dim, opt.embedding_dim)
        self.fc2 = nn.Linear(opt.embedding_dim, opt.projected_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        # Apply weight initialization
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z


class InvertibleNetBlock(nn.Module):
    """
    Defines a block within an Invertible Network consisting of a sequence of invertible operations.
    
    Args:
        input_dim (int): Dimension of the input features.
        blocks (int): Number of invertible blocks to stack.
    """
    def __init__(self, input_dim, blocks=16):
        super(InvertibleNetBlock, self).__init__()
        self.inn = SequenceINN(input_dim)
        for _ in range(blocks):
            self.inn.append(AllInOneBlock, subnet_constructor=nn.Linear)
            self.inn.append(PermuteRandom)

    def forward(self, x):
        x, _ = self.inn(x)
        return x


class SyntheticFeatureGenerator(nn.Module):
    """
    Generates synthetic features from noise and class attributes.
    
    Args:
        opt: Configuration options including sizes for input, hidden, and output layers.
    """
    def __init__(self, opt):
        super(SyntheticFeatureGenerator, self).__init__()

        self.fc1 = nn.Linear(opt.attribute_embedding_size + opt.noise_dim, opt.generator_hidden_dim)
        self.fc2 = nn.Linear(opt.generator_hidden_dim, opt.image_feature_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        # Apply invertible network block
        self.inn = InvertibleNetBlock(opt.image_feature_dim)
        # Apply weight initialization
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.inn(h)
        return h


class GANQualityCritic(nn.Module):
    """
    Critic network to evaluate the quality of generated features in relation to class attributes.
    
    Args:
        opt: Configuration options including sizes for input and hidden layers.
    """
    def __init__(self, opt):
        super(GANQualityCritic, self).__init__()
        self.fc1 = nn.Linear(opt.image_feature_dim + opt.attribute_embedding_size, opt.discriminator_hidden_dim)
        self.fc2 = nn.Linear(opt.discriminator_hidden_dim, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # Apply weight initialization
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class FeatureAttributeDiscriminator(nn.Module):
    """
    Discriminator network for distinguishing between real and synthetic feature-attribute pairs.
    
    Args:
        opt: Configuration options including sizes for input, hidden, and output layers.
    """
    def __init__(self, opt):
        super(FeatureAttributeDiscriminator, self).__init__()
        in_features = opt.embedding_dim + opt.attribute_embedding_size
        out_features = opt.comparator_network_dim
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        # Apply weight initialization
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h