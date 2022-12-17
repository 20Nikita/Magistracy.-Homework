import torch
import torch.optim as optim
import torch.nn as nn
from cod.dataloader import dataloader
from cod.train_alg import train_model
import cod.MyModels as models

def train(train_dirr,train_file,val_dirr,val_file,test_file,SIZE,batch_size,num_epochs,lr,my_rank):
    Dataset = "retinopatia"
    N_class = 5
    magistral = "efficientnet_b0"
#     magistral = "mobilenetv3_small_050"
    classification_criterion = nn.CrossEntropyLoss()

    train_dataloader = dataloader(SIZE, train_dirr, train_file, 'train', Dataset, N_class, batch_size = batch_size, shuffle = True, t1 = "resized_train/resized_train")
    val_dataloader = dataloader(SIZE, val_dirr, val_file, 'val', Dataset, N_class, batch_size = batch_size, shuffle = False, t1 = "resized_train/resized_train")
    test_dataloader = dataloader(SIZE, val_dirr, test_file, 'val', Dataset, N_class, batch_size = batch_size, shuffle = False, t1 = "resized_train/resized_train")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.CNN(N_class,magistral)
    model_ft = model_ft.to(device)

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
    test_out = train_model(model_ft, 
            classification_criterion, 
            optimizer_ft, 
            train_dataloader, 
            val_dataloader, 
            test_dataloader,
            batch_size, 
            num_epochs,
            my_rank)
    return test_out
