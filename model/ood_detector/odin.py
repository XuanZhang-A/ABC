import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms


class ODIN(nn.Module):
    def __init__(self, noiseMagnitude=1e-3, temperature=1e-2):
        super(ODIN, self).__init__()
        self.noiseMagnitude = noiseMagnitude
        self.temperature = temperature
        # not sure if this is the right way to normalize the gradient
        self.transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (63.0/255.0, 62.1/255.0, 66.7/255.0))])
    
    def get_gradient(self, inputs):
        # Get gradient after loss.backwards
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        gradient = self.transform(gradient)
        return gradient

    def cal_deviation(self, input, gradient, model):
        # Adding small perturbations to images that create in distribution
        inputs = torch.add(input.data, -1 * self.noiseMagnitude, gradient)
        outputs = model(Variable(inputs))
        outputs = outputs / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data[0]
        nnOutputs = nnOutputs - torch.max(nnOutputs)
        deviation = torch.exp(nnOutputs)/torch.sum(torch.exp(nnOutputs))

        return deviation

    def forward(self, inputs, model):
        gradient = self.get_gradient(inputs)
        deviation = self.cal_deviation(inputs, gradient, model)
        return deviation