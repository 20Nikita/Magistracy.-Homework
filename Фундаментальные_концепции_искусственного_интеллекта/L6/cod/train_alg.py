# -*- coding: utf-8 -*-
import time
import copy
import torch

def train_model(model, 
                classification_criterion, 
                optimizer, 
                train_dataloader, 
                val_dataloader, 
                test_dataloader,
                batch_size, 
                num_epochs=100,
                my_rank = 0,
                rezim = ['T', 'V']):
    # Запомнить время начала обучения

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_acc = 0         # Лучший покозатель модели
    
    for epoch in range(num_epochs):
        print('Rank {} epoch {}/{}'.format(my_rank, epoch + 1, num_epochs) + "\n")
        print('-' * 10 + "\n")
        
        # У каждой эпохи есть этап обучения и проверки
        for phase in rezim:
            if phase == 'T':
                dataloader = train_dataloader
                dataset_sizes = len(train_dataloader) * batch_size
                model.train()  # Установить модель в режим обучения
            elif phase == 'V':
                dataloader = val_dataloader
                dataset_sizes = len(val_dataloader) * batch_size
                model.eval()   #Установить модель в режим оценки
            
            # Обнуление параметров
            running_classification_loss = 0.0
            running_corrects = 0
            iiter = 0
            # Получать порции картинок и иx классов из датасета
            for inputs, classification_label in dataloader:
                # print(my_rank,iiter,len(dataloader) )    
                iiter+=1
                # считать все на видеокарте или ЦП
                inputs = inputs.to(device)
                classification_label = classification_label.to(device)
                # обнулить градиенты параметра
                optimizer.zero_grad()
                # forward
                # Пока градиент можно пощитать, шитать только на учимся
                with torch.set_grad_enabled(phase == 'T'):
                    # Проход картинок через модель
                    classification = model(inputs)
                    # Получить индексы максимальных элементов
                    _, preds = torch.max(classification, 1)
                    loss = classification_criterion(classification, classification_label.long())
                    # Если учимся
                    if phase == 'T':
                        # Вычислить градиенты
                        loss.backward()
                        # Обновить веса
                        optimizer.step()
                # Статистика
                running_classification_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == classification_label.data)# Колличество правильных ответов
            # Усреднить статистику
            epoch_classification_loss = running_classification_loss / dataset_sizes
            running_classification_loss/= dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            
            print('{}_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc))
                
            # Копироование весов успешной модели на вэйле
            if (phase == 'V') and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts_acc = copy.deepcopy(model.state_dict())
    
    del train_dataloader, val_dataloader
    model.load_state_dict(best_model_wts_acc)
    model.eval()
    test_out = []
    running_corrects = 0
    for inputs, classification_label in test_dataloader:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classification_label = classification_label.to(device)
        inputs = inputs.to(device)
        classification = model(inputs)
        _, preds = torch.max(classification, 1)
        test_out.append(preds)
        running_corrects += torch.sum(preds == classification_label.data)# Колличество правильных ответов
    dataset_sizes = len(test_dataloader) * batch_size
    epoch_acc = running_corrects / dataset_sizes
    print('T_Loss: None,  Acc: {:.4f}\n\n\n'.format(epoch_acc))
    del test_dataloader
    test_out = torch.stack(test_out)
    return test_out
