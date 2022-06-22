# Mimesis_torch

# open a new virtual env with python 3.8.5
# install requirements.txt

## To load the model

import config

import utils

test_model = utils.load_model('model_weights.pth',config.DEVICE,config.NUM_CLASSES)


# To get a prediction from model
from PIL import Image

image = Image.open(image_path)

# Create the preprocessing transformation here

prep = config.val_transforms(image)

# unsqueeze batch dimension

batch_image = prep.unsqueeze(0)

# Get prediction

output = torch.argmax(test_model(batch_image))
