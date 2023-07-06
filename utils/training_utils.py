from tqdm import tqdm
from time import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim import Optimizer
import numpy as np

from utils.data_utils import ImageDataset
from utils.log_save_utils import CheckpointManager, LogWriter
from models import *


def train(plain_model: Network, convex_model: Network, flow_model: FlowNetwork, 
          data_loader: DataLoader, all_data: ImageDataset,
          optimizer: Optimizer, criterion, scheduler: LRScheduler,
          epochs: int = 500, all_data_fraction: int = 0.8,
          save_log: bool = False, save_folder: str = r".snapshots"):
    
    if save_log:
        plain_cm = CheckpointManager(save_folder, "plain")
        convex_cm = CheckpointManager(save_folder, "convex")
        flow_cm = CheckpointManager(save_folder, "flow")

        training_lw = LogWriter(save_folder, "training")
    
    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            if save_log: start_time = time()

            for inputs, labels in data_loader:
                random_indices = np.random.random_integers(0, len(all_data) - 1, 
                                                           size = int(data_loader.batch_size * all_data_fraction))
                random_inputs = torch.tensor(all_data.data[random_indices])
                inputs = torch.concat((inputs, random_inputs), axis=0)

                outputs = plain_model(inputs)

                outputs = outputs[:len(labels)]
                loss = criterion(outputs, labels)

                outputs_flow = flow_model(inputs[:, :2])
                outputs_convex = convex_model(outputs_flow)

                outputs_convex = outputs_convex[:len(labels)]
                loss += criterion(outputs_convex, labels)

                # TODO Change to MSELoss
                if epoch > 200:
                    loss += torch.mean((outputs_convex - (outputs > 0.5).float()) ** 2)
                elif epoch > 30:
                    loss += torch.mean((outputs_convex - outputs) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                convex_model.step()

            
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)

            if save_log:
                training_lw.log(epoch, start_time, loss, 0, 0)
                progress.set_postfix(loss=float(loss))
                if epoch % 50:
                    plain_cm.save(plain_model, optimizer, epoch, loss)
                    convex_cm.save(convex_model, optimizer, epoch, loss)
                    flow_cm.save(flow_model, optimizer, epoch, loss)

    return plain_model, convex_model, flow_model