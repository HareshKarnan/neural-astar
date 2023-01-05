"""
Attempting to rewrite the training loop in pytorch lightning
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

import argparse
from termcolor import cprint

from neural_astar.utils.training import set_global_seeds, calc_metrics, visualize_results
from neural_astar.utils.data import create_dataloader
from neural_astar.planner import NeuralAstar, VanillaAstar

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import tensorboard as tb

from datetime import datetime


# create lightning data module
class MazeDataModule(pl.LightningDataModule):
    def __init__(self, filename, batch_size):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return create_dataloader(filename=self.filename, split="train", batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return create_dataloader(filename=self.filename, split="valid", batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return create_dataloader(filename=self.filename, split="test", batch_size=self.batch_size, shuffle=False)
    
# create lightning model
class NeuralAstarNetwork(pl.LightningModule):
    def __init__(self, encoder_arch):
        super().__init__()
        self.encoder_arch = encoder_arch
        self.neuralastar_model = NeuralAstar(encoder_arch=self.encoder_arch)
        
        self.vanillaastar_model = VanillaAstar()
        # set vanilla astar to eval mode
        self.vanillaastar_model.eval()
        
        self.loss = torch.nn.L1Loss()
    
    def forward(self, map_designs, start_maps, goal_maps):
        return self.neuralastar_model(map_designs, start_maps, goal_maps)
    
    def training_step(self, batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = batch
        
        planner_outputs = self.forward(map_designs, start_maps, goal_maps)
        
        loss = self.loss(planner_outputs.histories, opt_trajs)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = batch
        planner_outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = self.loss(planner_outputs.histories, opt_trajs)
        
        # calculate metrics
        with torch.no_grad():
            vanilla_planner_outputs = self.vanillaastar_model(map_designs, start_maps, goal_maps)
            metrics = calc_metrics(planner_outputs, vanilla_planner_outputs)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_p_opt", metrics.p_opt, on_step=True, on_epoch=True)
        self.log("val_p_exp", metrics.p_exp, on_step=True, on_epoch=True)
        self.log("val_h_mean", metrics.h_mean, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = batch
        planner_outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = self.loss(planner_outputs.histories, opt_trajs)
        
        # calculate metrics
        with torch.no_grad():
            vanilla_planner_outputs = self.vanillaastar_model(map_designs, start_maps, goal_maps)
            metrics = calc_metrics(planner_outputs, vanilla_planner_outputs)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_p_opt", metrics.p_opt, on_step=True, on_epoch=True)
        self.log("test_p_exp", metrics.p_exp, on_step=True, on_epoch=True)
        self.log("test_h_mean", metrics.h_mean, on_step=True, on_epoch=True)
        return {"test_loss": loss}
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        with torch.no_grad():
            map_designs, start_maps, goal_maps, opt_trajs = batch
            
            va_output = self.vanillaastar_model(map_designs, start_maps, goal_maps)
            na_output = self.neuralastar_model(map_designs, start_maps, goal_maps)
            
            va_results = visualize_results(map_designs, va_output)
            na_results = visualize_results(map_designs, na_output)
        
        self.logger.experiment.add_images("vanilla_A_star", va_results, global_step=self.current_epoch, dataformats="HWC")
        self.logger.experiment.add_images("neural_A_star", na_results, global_step=self.current_epoch, dataformats="HWC")
    
    def configure_optimizers(self):
        return torch.optim.RMSprop(self.neuralastar_model.parameters(), lr=1e-3)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", "-d", type=str, default="data/mazes_032_moore_c8.npz")
    parser.add_argument("--logdir", "-l", type=str, default="log")
    parser.add_argument("--encoder_arch", "-e", type=str, default="CNN", choices=["CNN", "Unet"])
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--num_epochs", "-n", type=int, default=50)
    parser.add_argument("--num_gpus", "-g", type=int, default=1)
    parser.add_argument("--seed", "-s", type=int, default=1234)
    args = parser.parse_args()
    
    # set the global seed
    set_global_seeds(args.seed)
    
    # create maze datamodule
    dm = MazeDataModule(args.data_npz, args.batch_size)
    
    # setup the model
    model = NeuralAstarNetwork(args.encoder_arch)
    
    # callbacks for PyTorch Lightning
    early_stopping = EarlyStopping(monitor="val_loss", patience=100, verbose=True)
    model_checkpoint = ModelCheckpoint(dirpath='trained_models/',
                                       filename='neuralastar-{epoch:02d}-{val_loss:.2f}',
                                       monitor='val_loss', mode='min')
    
    # PyTorch Lightning trainer
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.num_epochs,
        callbacks=[early_stopping, model_checkpoint],
        logger=pl.loggers.TensorBoardLogger('lightning_logs/'),
        strategy="ddp",
        # num_sanity_val_steps=0,
    )    
    
    # train the model
    cprint('Training the model!', 'yellow', attrs=['bold', 'blink'])
    trainer.fit(model, dm)
    
    # test the model
    trainer.test(model, dm)
    
    # save the model
    torch.save(model.state_dict(), 'trained_models/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_final_' + '.pt')
    cprint('model trained and saved!!!', 'green', attrs=['bold'])
    
    
    
    
    
    