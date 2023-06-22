from fastai import *
from fastai.vision.all import *
from torchvision.models import * 
from fastai.callback.hook import *
import cv2
from received_image import ReceivedImage
from PIL import Image
from torchvision.transforms import ToTensor
from fastai.vision.core import TensorImage
from random import randint
from fastai.callback.hook import *
from random import randint
import matplotlib.pyplot as plt
import cv2
from fastai.callback.hook import *
import torch
from fastai.vision.core import TensorImage
import matplotlib.pyplot as plt
import numpy as np
import fastai
from PIL import Image
import torch.nn.functional as F


class Detector:

    def __init__(self, model_path = None, arch = densenet201, bs = 10, sz = 512):
        self.__arhitecture = arch
        self.__bs = bs
        self.__sz = sz
        self.__learner = load_learner(model_path)

    def predict(self, image):
        predicted_class = self.__learner.predict(image)
        return predicted_class[-1].tolist()
    
    def predict_with_model(self, image):
        image = TensorImage(image).permute((2, 0, 1)).float() / 255
        image = image.to(self.__learner.dls.device)  
        raw_output = self.__learner.model(image)
        probabilities = F.softmax(raw_output, dim=1)
        return probabilities, raw_output

    def predict_images(self, images: list[ReceivedImage]):
        predictions = [self.__learner.predict(x.image) for x in images]
        return [[p[-1].tolist(), t.position] for p, t in zip(predictions, images)]
    
    # def generate_heatmap(self, image):
    #     # Create a hook to extract the activations from the final convolutional layer
    #     hook = hook_output(self.__learner.model[-1])
    #     image = Image.fromarray(image)
    #     image = ToTensor()(image)
    #     # Pass the image through the model to get the activations
    #     self.__learner.model.eval()
    #     with torch.no_grad():
    #         _ = self.__learner.model(image.unsqueeze(0))

    #     activations = hook.stored[0].cpu()
    #     avg_activations = torch.mean(activations, dim=0)

    #     heatmap = avg_activations.numpy()
    #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize the heatmap
    #     heatmap = np.uint8(255 * heatmap)

    #     return heatmap


    # def plot_heatmap(self, learner, numpy_image, classes=['Negative', 'Tumor']):
    #     def hooked_backward(m, oneBatch, cat):
    #         with hook_output(m[0]) as hook_a: 
    #             with hook_output(m[0], grad=True) as hook_g:
    #                 preds = m(oneBatch)
    #                 preds[0, int(cat)].backward()
    #         return hook_a, hook_g

    #     def getHeatmap(tensorImg, cl):
    #         m = learner.model.eval()
    #         infer_dl = learner.dls.test_dl([tensorImg])
    #         oneBatch, = next(iter(infer_dl))
    #         oneBatch_im = Image.fromarray(np.array(oneBatch[0]).astype(np.uint8))
    #         cvIm = np.array(oneBatch_im).astype(np.uint8)
    #         hook_a, hook_g = hooked_backward(m, oneBatch, cl)
    #         acts = hook_a.stored[0].cpu()
    #         grad = hook_g.stored[0][0].cpu()
    #         grad_chan = grad.mean(1).mean(1)
    #         mult = (acts * grad_chan[..., None, None]).mean(0)
    #         return mult, cvIm

    #     tensorImg = tensor(numpy_image)
    #     cl = randint(0, len(classes) - 1)  
    #     act, im = getHeatmap(tensorImg, cl)
    #     fig, ax = plt.subplots()
    #     H, W = im.shape[:2]
    #     ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #     im = ax.imshow(
    #         act, alpha=0.5, extent=(0, H, W, 0), interpolation='bilinear', cmap='inferno'
    #     )
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(f'Heatmap for {classes[cl]}')
    #     cbar = plt.colorbar(im)
    #     plt.show()

    
    def generate_heatmap(self, image):
        return self.plot_heatmap(self.__learner, image)

    def plot_heatmap(self, learner, np_image):
        image = TensorImage(np_image).permute((2, 0, 1)).float() / 255
        # image = learner.dls.after_batch.normalize(image.unsqueeze(0))
        image = image.to(learner.dls.device)  # move image to the device learner is using
        learner.model.eval()
        target_layer = learner.model[0][-1]

        # hook the activations and gradients
        with hook_output(target_layer) as hook_a:
            with hook_output(target_layer, grad=True) as hook_g:
                preds = learner.model(image.unsqueeze(0))[0]
                class_index = preds.argmax() 
                preds[class_index].backward()

        # get the activations and gradients
        acts = hook_a.stored[0].cpu()
        grads = hook_g.stored[0][0].cpu()

        # global average pooling
        weights = torch.mean(grads, dim=[1, 2])

        # multiply each channel in the feature map by corresponding weight
        saliency_map = (weights[:, None, None] * acts).sum(dim=0)

        # ReLU on heatmap
        saliency_map = np.maximum(saliency_map, 0)

        # Normalize saliency_map between new_min and new_max
        new_min, new_max = 0.5, 1.0  # adjust these values as needed
        saliency_map = new_min + (saliency_map - saliency_map.min()) * (new_max - new_min) / (saliency_map.max() - saliency_map.min() + 1e-8)

        # convert to numpy for visualization
        saliency_map = saliency_map.detach().numpy()  # convert to numpy

        # convert to heatmap format
        heatmap = cv2.applyColorMap(np.uint8(255*saliency_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # resize the heatmap to the size of the input image
        heatmap_resized = cv2.resize(heatmap, (image.shape[2], image.shape[1]))

        # overlay the heatmap on the original image
        cam = heatmap_resized + np.float32(image.permute(1,2,0).cpu().numpy())  # move to CPU before converting to numpy
        cam = cam / np.max(cam)

        return cam




