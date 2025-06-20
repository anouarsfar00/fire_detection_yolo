import os, yaml, csv
from ultralytics import YOLO
import torch.multiprocessing

os.makedirs('reports', exist_ok=True)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # recommandé pour Windows
    
    # Lire les paramètres
    with open(r"params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Charger un modèle pré-entraîné
    pre_trained_model = YOLO(params['model_type'])
    
    # Entraîner le modèle
    model = pre_trained_model.train(
        data='C:\\Users\\SPIRIT\\Desktop\\master_anouar\\master_anouar\\data\\Fire_dataset\\data.yaml',
        imgsz=params['imgsz'],
        batch=params['batch'],
        epochs=params['epochs'],
        optimizer=params['optimizer'],
        lr0=params['lr0'],
        seed=params['seed'],
        pretrained=params['pretrained'],
        name=params['name'],
        project='models',
        exist_ok=True,
        device=0,
        # callbacks=[]  # désactive MLflow si nécessaire
    )
    
    # === AJOUT : Sauvegarder les paramètres d'entraînement ===
    train_params = {
        'model_type': params['model_type'],
        'imgsz': params['imgsz'],
        'batch': params['batch'],
        'epochs': params['epochs'],
        'optimizer': params['optimizer'],
        'lr0': params['lr0'],
        'seed': params['seed'],
        'pretrained': params['pretrained'],
        'name': params['name']
    }
    
    with open('reports/train_params.yaml', 'w') as f:
        yaml.dump(train_params, f)
    
    # === AJOUT : Sauvegarder les métriques d'entraînement ===
    # Les résultats YOLO sont généralement dans model.trainer.metrics
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'metrics'):
        metrics = model.trainer.metrics
        
        # Créer un fichier CSV avec les métriques
        with open('reports/train_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            
            # Extraire les métriques disponibles
            if hasattr(metrics, 'results_dict'):
                for key, value in metrics.results_dict.items():
                    writer.writerow([key, value])
            else:
                # Fallback : créer des métriques basiques
                writer.writerow(['epochs_completed', params['epochs']])
                writer.writerow(['batch_size', params['batch']])
                writer.writerow(['image_size', params['imgsz']])
    else:
        # Fallback : créer un fichier CSV minimal
        with open('reports/train_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['epochs_completed', params['epochs']])
            writer.writerow(['batch_size', params['batch']])
            writer.writerow(['image_size', params['imgsz']])
    
    print("Entraînement terminé. Fichiers de rapports générés dans le dossier 'reports/'.")