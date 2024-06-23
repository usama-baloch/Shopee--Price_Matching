train_images_path = ''
train_csv_path = ''
test_images_path = ''
test_csv_path = ''

IMG_SIZE = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

Epochs = 15
BATCH_SIZE = 8
BATCH_SIZE_VALID = 12

num_workers = 4 
device = "cuda"
seed = 2020

classes = 11014 # number of unique classes in the dataset
Scale = 30.0
Margin = 0.5

model_name = 'eca_nfnet_l0'
model_path = ''

FC_DIM = 512
SCHEDULER_PARAMS = {
    "lr_start": 1e-5,
    "lr_max": 1e-5 * 32,
    "lr_min": 1e-6,
    "lr_ramp_ep": 5,
    "lr_sus_ep": 0,
    "lr_decay": 0.8
}