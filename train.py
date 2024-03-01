import pytorch_lightning as pl
import yaml
from models.model import TyNet
from utils.utils import init_weights
import argparse
from data.coco_dataset import CustomDataset, collater
from data.augmentations import get_augmentations
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.loss import FocalLoss
from models.utils import get_optimizer, get_scheduler
from models.detector import Detector
import torch 


if __name__ == '__main__':

    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='training.yaml', help='training config file')
    parser.add_argument('--dataset_cfg', type=str, default='coco.yml', help='training config file')
    
    args = parser.parse_args()

    opt = args.train_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)

    dataset_opt = args.dataset_cfg
    with open(dataset_opt, 'r') as config:
        dataset_opt = yaml.safe_load(config)

    model = TyNet(num_classes=len(dataset_opt['obj_list']),
                  ratios=eval(dataset_opt['anchors_ratios']), 
                  scales=eval(dataset_opt['anchors_scales']))

    init_weights(model)
    model = model.cuda()

    augmentations = get_augmentations(opt)

    training_params = {'batch_size': opt['training']['batch_size'],
                        'shuffle': opt['training']['shuffle'],
                        'drop_last': opt['training']['drop_last'],
                        'collate_fn': collater,
                        'num_workers': opt['training']['num_workers']}

    val_params = {'batch_size': 1,
                    'shuffle': opt['validation']['shuffle'],
                    'drop_last': opt['validation']['drop_last'],
                    'collate_fn': collater,
                    'num_workers': opt['validation']['num_workers']}
    
    train_dataset = CustomDataset(image_path=opt['training']['image_path'], 
                        annotation_path=opt['training']['annotation_path'], 
                        image_size=opt['training']['image_size'], 
                        normalize=opt['training']['normalize'],
                        augmentations=augmentations)
    
    val_dataset = CustomDataset(image_path=opt['validation']['image_path'], 
                        annotation_path=opt['validation']['annotation_path'], 
                        image_size=opt['training']['image_size'], 
                        normalize=opt['training']['normalize'], 
                        augmentations=None)

    
    train_loader = DataLoader(train_dataset, **training_params)

    val_loader = DataLoader(val_dataset, **val_params)

    logger = TensorBoardLogger("tb_logs", name="my_model")

    loss_fn = FocalLoss()

    optimizer = get_optimizer(opt['training'], model)
    scheduler = get_scheduler(opt['training'], optimizer, len(train_loader))
    
    detector = Detector(model=model, scheduler=scheduler, optimizer=optimizer, loss=loss_fn)

    trainer = pl.Trainer(gpus=1, logger=logger, check_val_every_n_epoch=opt['training']['val_frequency'], max_epochs=opt['training']['epochs'])
    
    trainer.fit(model=detector, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
                
    # save model
    torch.save(detector.model.state_dict(), 'TyNet.pth')
    
    # save to onnx
    model.eval()

    # Create a dummy input tensor of the correct size
    x = torch.randn(1, 3, 480, 480)

    # Specify input and output names
    input_names = ["input"]
    output_names = ["output", "regression", "classification", "anchors"]

    # Set dynamic axes
    dynamic_axes = {"input" : {0 : "batch_size"}, "output" : {0 : "batch_size"}, "regression" : {0 : "batch_size"}, "classification" : {0 : "batch_size"}, "anchors" : {0 : "batch_size"}}

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "TyNet.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = input_names, # the model's input names
                    output_names = output_names, # the model's output names
                    dynamic_axes=dynamic_axes) # variable length axes
