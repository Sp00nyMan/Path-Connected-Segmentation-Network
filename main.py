import torch
from utils.data_utils import ImageDataset
from torch.utils.data import DataLoader

from models import Network, FlowNetwork
from torch.optim import Adam
from torch.nn import BCELoss
from utils.training_utils import train

def train_main(filename: str):    
    original_image = f"data/{filename}.png"
    image_fore = f"data/{filename}_fore.png"
    image_back = f"data/{filename}_back.png"

    criterion = BCELoss()
    batch_size = 1500

    dataset = ImageDataset(original_image, image_fore, image_back)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    all_dataset = ImageDataset(original_image, None, None, train=False)

    convex = Network(2, hidden_neurons=80, convex=True)
    optimizer = Adam(convex.parameters(), lr=1e-3)

    train(None, convex, None, data_loader, all_dataset, optimizer, criterion,
          f"{filename}_convex", teacher=False, pc=False)
    
    teacher = Network(5)
    convex = Network(2, hidden_neurons=80, convex=True)
    optimizer = Adam(list(teacher.parameters()) + 
                     list(convex.parameters()), 
                     lr=1e-3)

    train(teacher, convex, None, data_loader, all_dataset, optimizer, criterion,
          f"{filename}_convex_teacher", pc=False)
    
    convex = Network(2, hidden_neurons=80, convex=True)
    flow_model = FlowNetwork(2, 50)
    optimizer = Adam(list(flow_model.parameters()) + 
                    list(convex.parameters()), 
                    lr=1e-3)

    train(None, convex, flow_model, data_loader, all_dataset, optimizer, criterion,
          f"{filename}_pc", teacher=False)
    
    teacher = Network(5)
    convex = Network(2, hidden_neurons=80, convex=True)
    flow_model = FlowNetwork(2, 50)
    optimizer = Adam(list(teacher.parameters()) +
                     list(flow_model.parameters()) + 
                    list(convex.parameters()), 
                    lr=1e-3)

    train(teacher, convex, flow_model, data_loader, all_dataset, optimizer, criterion,
          f"{filename}_pc_teacher")


def main():
    train_main("apples")
    train_main("mandarins")