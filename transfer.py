from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.animation as animation


import copy

from fastprogress import master_bar, progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    print("The program will run extremly slowly on CPU")


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = tensor(mean).view(-1, 1, 1).to(device)
        self.std = tensor(std).view(-1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target: tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super().__init__()
        self.target = gram_matrix(target_features).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class StyleTransferer:

    class Loader:

        def __init__(self, shape) -> None:
            self.loader = transforms.Compose([
                transforms.Resize(shape),
                transforms.ToTensor()
            ])

            self.unloader = transforms.ToPILImage()

        def __call__(self, img_name):
            image = Image.open(img_name)
            image = self.loader(image).unsqueeze(0)
            return image.to(device, torch.float)

        def to_show(self, img):
            img = img.clone()
            img = img.squeeze(0)
            return self.unloader(img)

        def imshow(self, img, figsize=None, title=None):
            img = img.clone()
            img = img.squeeze(0)
            img = self.unloader(img)

            plt.figure(figsize=figsize)
            plt.imshow(img)

            if title is not None:
                plt.title(title)

    def __init__(self,
                 shape,
                 content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']) -> None:
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.change_image_shape(shape)

    def get_model(self, content_image, style_image):

        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        norm = Normalization(norm_mean, norm_std).to(device)
        content_losses = []
        style_losses = []

        model = nn.Sequential(norm).to(device)

        # if doesnt work, try adding deepcopy here

        counter = 0

        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                counter += 1
                name = 'conv_{}'.format(counter)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(counter)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(counter)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(counter)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(
                    'content_loss_{}'.format(counter), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target = model(style_image).detach()
                style_loss = StyleLoss(target)
                model.add_module('style_loss_{}'.format(counter), style_loss)
                style_losses.append(style_loss)

        # removing all the layers we dont need
        for i in range(len(model)-1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:i+1]

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def change_image_shape(self, shape):
        self.loader = StyleTransferer.Loader(shape)

    def apply(self, source_name, content_name, n_steps=300, c_content=1, c_style=2e3, show_animation=False, save_animation=None):

        content = self.loader(content_name)
        source = self.loader(source_name)

        self.get_model(content, source)

        optimizer = optim.LBFGS([content.requires_grad_()])

        def closure():
            content.data.clamp_(0, 1)
            optimizer.zero_grad()
            self.model(content)

            style_score = 0
            content_score = 0

            for sl in self.style_losses:
                style_score += sl.loss

            for cl in self.content_losses:
                content_score += cl.loss

            style_score *= c_style
            content_score *= c_content

            loss = style_score + content_score
            loss.backward()

            return loss

        # drawing
        ims = []
        ax = None
        fig = None
        im_ani = None
        show_animation = True if save_animation is not None else show_animation

        if show_animation:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                                 xlim=(-100, 100), ylim=(-100, 100))

        for _ in progress_bar(range(n_steps)):
            optimizer.step(closure)
            ims.append([ax.imshow(self.loader.to_show(content))])

        if show_animation:
            im_ani = animation.ArtistAnimation(fig, ims, interval=120, repeat_delay=1500,
                                               blit=True)

        if save_animation is not None and show_animation:
            im_ani.save(save_animation + '.mp4', metadata={'artist': 'Guido'})

        self.loader.imshow(content, figsize=(10, 10))
        return content


m = StyleTransferer((512, 512))

m.apply('./picasso.jpg', './image.jpg')
plt.show()
