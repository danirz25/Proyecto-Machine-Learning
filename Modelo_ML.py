# Proyecto del programa delfin
#Daniel Patiño Ruiz

#Libraries de Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
#Para visualizaciones
import matplotlib.pyplot as plt
import numpy as np
#Carga de datos y transformations
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
#Para las metricas
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
import seaborn as sns 
#para las rutas
import os
from sklearn.decomposition import PCA

#Define si los calculos se hacen en CPU o GPU
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
data_dir = "D:/Escuela/Pro_Delfin/archive/chest_xray"

#Transformar las imágenes a 224x224 y normalización
#DenseNet trabaja con imágenes 224x224.
# dos conjuntos de transformacion: train y val  

transform = {
    'train': transforms.Compose([
       # transforms.Resize((224, 224)), #redimension                                      VERSION 1

       # transforms.Resize(256),  # reduce el lado más largo a 256 manteniendo proporción  VERSION 2
       # transforms.CenterCrop(224),  # recorta el centro a 224x224                        VERSION 2

        transforms.RandomRotation(10),           # Gira poquito la imagen                   VERSION 3
        transforms.RandomResizedCrop(224),       # Recorte aleatorio                        VERSION3
        transforms.ColorJitter(brightness=0.1),  # Cambios de brillo                        VERSION 3


        transforms.RandomHorizontalFlip(),  # aumento de datos
        transforms.ToTensor(), #convierte en tensor
        transforms.Normalize([0.485], [0.229])  # normalizacion de un canal
    ]),
    'val': transforms.Compose([
       #transforms.Resize((224, 224)), #Lo mismo que arriba xd                             VERSION 1
        transforms.Resize(256), #Lo mismo de arriba                                         VERSION 2
        transforms.CenterCrop(224),                                                        #VERSION 2
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
}

#Cargar el dataset
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform['train']),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform['val']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform['val'])  # Misma transformación que val
}

#dataloader
#Creacion de lotes de datos para train y val
dataloaders = {   
    phase: DataLoader(image_datasets[phase], batch_size=32, shuffle=True) #Lotes de 32 batch y aleatorios en c. epoca
    for phase in ['train', 'val', 'test'] #para las 3 carpetas
}
dataset_sizes = {phase: len(image_datasets[phase]) for phase in ['train', 'val', 'test']} #Total de imagenes por epoca
class_names = image_datasets['train'].classes  # Otras clases: NORMAL y NEUMONIA (carpetas)

#Modelo DENSENET-121 
model = models.densenet121(pretrained=True) #Carga del modelo
num_features = model.classifier.in_features #Obtiene el Num de entradas de la capa final
model.classifier = nn.Linear(num_features, 1)  # salida binaria 
model = model.to(device) #Mueve a CPU o GPU dependiendo disponibilidad

# Extraer capa de embeddings
feature_extractor = nn.Sequential(*(list(model.children())[:-1]))

#Funcipn de perdida
criterion = nn.BCEWithLogitsLoss()

#Optimizador SGD con momentum
#taza de aprendizaje  = 0.01
# momentum = 0.9
#Si se sube el momentum baja el lr para compensar y equilibrar

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)    #Optimizador version 1
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)   #Optimizador v 2

# Listas para graficar luego
total_train_loss, total_val_loss = [], []
total_train_acc, total_val_acc = [], []


#Para poder entrenar el modelo
#Funcion de entrenamiento del modelo, con 5 epochs o epocas
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"\n Epoca {epoch+1}/{num_epochs}") #Num de epoca actual

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval() #modo de entrenamiento
            running_loss = 0.0  #Acumulador de perdida
            running_corrects = 0 #Acumulador de aciertos

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) #Mover imagenes al GPU / CPU
                labels = labels.to(device).float().unsqueeze(1)  # Convertir batch a batch,1
                optimizer.zero_grad() #Limpia gradientes acumuladosh

                with torch.set_grad_enabled(phase == 'train'): #Calcula gradientes
                    outputs = model(inputs)  #Prediccion del modelo
                    loss = criterion(outputs, labels) #Calcula la perdida 
                    preds = (torch.sigmoid(outputs) > 0.5).float()  #Logits a 0 o 1, o sea, clasif binaria  umbral de 0.5

                    if phase == 'train':
                        loss.backward() #backpropragation (ajuste de parametros)
                        optimizer.step() #Actualiza parametros

                running_loss += loss.item() * inputs.size(0) #Perdida total por num de ejemplos
                running_corrects += torch.sum(preds == labels) #acumula aciertos

            epoch_loss = running_loss / dataset_sizes[phase] #perdida promedio
            epoch_acc = running_corrects.double() / dataset_sizes[phase] #precision promedio
            print(f"{phase.capitalize()} - Perdida: {epoch_loss:.4f}, Precision: {epoch_acc:.4f}") #se imprimen las metricas obtenidas

            if phase == 'train':
                total_train_loss.append(epoch_loss)
                total_train_acc.append(epoch_acc.cpu().item())
            else:
                total_val_loss.append(epoch_loss)
                total_val_acc.append(epoch_acc.cpu().item())

    # Graficar perdidas y precision por epoca
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, total_train_loss, label='Train Loss')
    plt.plot(epochs, total_val_loss, label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida por época')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, total_train_acc, label='Train Acc')
    plt.plot(epochs, total_val_acc, label='Val Acc')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión por época')
    plt.legend()
    plt.tight_layout()
    plt.show()

#Evaluar el modelo con metricas
#Tmb evalua con curva ROC
def evaluate_model(model, dataloader): 
    model.eval()
    all_labels, all_probs, all_inputs, all_embeddings = [], [], [], []

    with torch.no_grad():  #sin gradientes, sin entrenar 
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            features = model.features(inputs)
            pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(inputs.size(0), -1)
            outputs = model.classifier(pooled)
            probs = torch.sigmoid(outputs)

            all_embeddings.extend(pooled.cpu().numpy())
            all_inputs.extend(inputs.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            """
            inputs = inputs.to(device)  #mover imgs a cpu o gpu 
            labels = labels.to(device).float().unsqueeze(1) #loo mismo de arriba 
            outputs = model(inputs)
            probs = torch.sigmoid(outputs) #probabilidades de 0 a 1

            all_inputs.extend(inputs.cpu())
            all_labels.extend(labels.cpu().numpy()) #etiquetas en cpu
            all_probs.extend(probs.cpu().numpy()) #probabs en cpu
            """
    #preparacion de arrays para metricas
    y_true = np.array(all_labels).flatten() #etiquetas
    y_prob = np.array(all_probs).flatten() #probabs
    y_pred = (y_prob > 0.5).astype(int) #binarizacion de la probabilifad, 0 o 1 

    #Metricas
    acc = accuracy_score(y_true, y_pred) #Exactitud 
    f1 = f1_score(y_true, y_pred) #Balance entre precisión y recall
    auc = roc_auc_score(y_true, y_prob) #ROC

    print(f"\n Precision: {acc:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}") 

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    labels = ['NORMAL', 'NEUMONIA']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.title('Matriz de Confusión')
    plt.show()

#Acá va a ir las gráficas para medir las métricas y exitos 
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob) #falsos + reales positivos
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}') #curva de modelo 
    plt.plot([0, 1], [0, 1], linestyle='--') #linea xd
    plt.xlabel('Falsos positivos')
    plt.ylabel('Reales positivos')
    plt.title('Curva ROC')
    plt.legend() #Leyenda como rango
    plt.grid(True) #Activa la cuadrícula
    plt.show() #ploteo de grafica

    # Esto es para mostrar imagenes de cuando el modelo detecta neumonia
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    idx = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and idx < 4:
            img = all_inputs[i].squeeze().numpy()
            axs[idx // 2, idx % 2].imshow(img[0] if img.ndim == 3 else img, cmap='gray')
            axs[idx // 2, idx % 2].set_title(f"Pred: NEUMONIA | Real: {'NEUMONIA' if y_true[i] == 1 else 'NORMAL'}")
            axs[idx // 2, idx % 2].axis('off')
            idx += 1
    plt.tight_layout()
    plt.show()

# Visualización de embeddings en 2D
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(all_embeddings)
    plt.figure(figsize=(8,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=y_true, cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Clase (0=NORMAL, 1=NEUMONIA)')
    plt.title('Visualización PCA de Embeddings')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.show()

#Execuxion de los datos de prueba una vez acabado el entrenamiento
if __name__ == "__main__": #Solo se ejecuta si lo corro directamente 
    train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5) #entrenamiento y val 5 veces
    evaluate_model(model, dataloaders['test']) #prueba de fuego




#No olvidar para GIT

"""
git status
git add . 
git commit -m "texto xd"
git push origin main
"""




