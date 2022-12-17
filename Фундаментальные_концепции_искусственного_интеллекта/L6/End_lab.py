
# cd C:\Users\Admin\Desktop\Универ\DZ\Фундаментальные_концепции_искусственного_интеллекта\Новая папка
# mpiexec -np 4 python End_lab.py

from mpi4py import MPI
from cod.train import train
from cod.dataloader import dataloader
import torch

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()

    dirr = "../data/Diabetic_Retinopathy"
    file = "trainLabels_cropped.csv"
    if my_rank ==0:
        input = open(dirr + "/" + file, 'r')
        shapca = input.readline()

        output = []
        for i in range(p):
            output.append(open(dirr + "/" + file.split(".")[0]+"_"+str(i)+"_train.csv", 'w'))
            output[-1].write(shapca)
            output.append(open(dirr + "/" + file.split(".")[0]+"_"+str(i)+"_val.csv", 'w'))
            output[-1].write(shapca)
        output.append(open(dirr + "/" + file.split(".")[0]+"_test.csv", 'w'))
        i=0
        for lite in input:
            ind = i % (p*2 + 1)
            output[ind].write(lite)
            i+=1
        for procid in range(1,p):
            comm.send("message",dest=procid)
    else:
        comm.recv(source=0)

    SIZE = 112
    train_dirr = val_dirr = dirr
    train_file = file.split(".")[0]+"_"+str(my_rank)+"_train.csv"
    val_file = file.split(".")[0]+"_"+str(my_rank)+"_val.csv"
    test_file = file.split(".")[0]+"_test.csv"
    N_class = 5
    batch_size = 50
    num_epochs = 100
    lr = 0.0001

    test_out = train(train_dirr,train_file,val_dirr,val_file,test_file,SIZE,batch_size,num_epochs,lr,my_rank)
    
    
    multi_out = []
    multi_out.append(test_out)
    if my_rank !=0:
        comm.send(test_out,dest=0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for procid in range(1,p):
            message = comm.recv(source=procid)
            multi_out.append(message)
        multi_out = torch.stack(multi_out)
        preds = torch.mode(multi_out, 0)[0].to(device)

        test_dataloader = dataloader(SIZE, val_dirr, test_file, 'val', "retinopatia", N_class, batch_size = batch_size, shuffle = False, t1 = "resized_train/resized_train")

        running_corrects = 0
        i=0
        for inputs, classification_label in test_dataloader:
            classification_label = classification_label.to(device)
            running_corrects += torch.sum(preds[i] == classification_label.data)
            i+=1
        dataset_sizes = len(test_dataloader) * batch_size
        epoch_acc = running_corrects / dataset_sizes
        print('-' * 10 + "\n")
        print('General Acc: {:.4f}'.format(epoch_acc))

