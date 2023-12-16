import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from einops import rearrange
from PIL import Image


def upscaling(tensor_img):
    transforms = T.ToPILImage()

    pil_image = transforms(tensor_img)

    resized_image = pil_image.resize((256, 256), resample=Image.BOX)

    return resized_image


def tensor2gomuboard(tensor, nrow, ncol, softmax=False, scale=1.0):
    if softmax:
        tensor = F.softmax(tensor.view(-1)).view(1, nrow, ncol).cpu().detach()

        return upscaling(tensor * scale)

    tensor = tensor.view(nrow, ncol)
    img = torch.zeros((nrow, ncol, 3))

    # white
    WHITE = torch.tensor([234., 232., 234.])/255
    # black
    BLACK = torch.tensor([94., 93., 94.])/255
    # blue
    BLUE = torch.tensor([208, 239, 255])/255

    img[tensor==1] = WHITE
    img[tensor==-1] = BLACK
    img[tensor==2] = BLUE

    img = rearrange(img, "h w c -> c h w")

    return upscaling(img)
