import torch

BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 1000 # Resize the image for training and transforms.
NUM_EPOCHS = 55 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_IMG = 'data/train_images'
TRAIN_ANNOT = 'data/train_annots'
# Validation images and XML files directory.
VALID_IMG = 'data/valid_images'
VALID_ANNOT = 'data/valid_annots'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__',
    'building',
    'ship',
    'vehicle',
    'prefabricated-house',
    'well',
    'cable-tower',
    'pool',
    'landslide',
    'cultivation-mesh-cage',
    'quarry'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'