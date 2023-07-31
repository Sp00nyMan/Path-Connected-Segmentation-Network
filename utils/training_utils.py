from tqdm import tqdm
from time import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from utils.data_utils import ImageDataset
from utils.log_save_utils import CheckpointManager, LogWriter
from models import *


def train(plain_model: Network, convex_model: Network, flow_model: FlowNetwork, 
          data_loader: DataLoader, all_data: ImageDataset,
          optimizer: Optimizer, criterion,
          name: str, 
          epochs: int = 300, all_data_fraction: int = 0.8,
          save_log: bool = True, save_folder: str = r".snapshots", 
          teacher=True, pc=True):
    
    if flow_model is not None and convex_model is None:
        raise ValueError("Training the flow model doesn't make sense without the convex model")
    
    save_folder += "/" + name

    if save_log:
        if teacher:
            plain_cm = CheckpointManager(save_folder, "plain")
        convex_cm = CheckpointManager(save_folder, "convex")
        if pc:
            flow_cm = CheckpointManager(save_folder, "flow")

    training_lw = LogWriter(save_folder)
    
    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            start_time = time()

            for inputs, labels in data_loader:
                random_indices = np.random.random_integers(0, len(all_data) - 1, 
                                                           size = int(data_loader.batch_size * all_data_fraction))
                random_inputs = torch.tensor(all_data.data[random_indices])
                inputs = torch.concat((inputs, random_inputs), axis=0)

                if teacher:
                    outputs = plain_model(inputs)

                    outputs = outputs[:len(labels)]
                    loss = criterion(outputs, labels)

                if pc:
                    outputs_flow = flow_model(inputs[:, :2])
                outputs_convex = convex_model(outputs_flow if pc else inputs[:, :2])

                outputs_convex = outputs_convex[:len(labels)]
                if teacher:
                    loss += criterion(outputs_convex, labels)
                else:
                    loss = criterion(outputs_convex, labels)

                # TODO Change to MSELoss
                if teacher:
                    if epoch > 200:
                        loss += torch.mean((outputs_convex - outputs) ** 2)
                    elif epoch > 30:
                        loss += torch.mean((outputs_convex - (outputs > 0.5).float()) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                convex_model.step()
            
            training_lw.log(epoch, start_time, loss, 0, 0)

            if save_log:
                progress.set_postfix(loss=float(loss))
                if teacher:
                    plain_cm.save(plain_model, optimizer, epoch, loss)
                convex_cm.save(convex_model, optimizer, epoch, loss)
                if pc:
                    flow_cm.save(flow_model, optimizer, epoch, loss)

    return plain_model, convex_model, flow_model