import pickle
import pandas as pd
import torch
from src.model import MyNN
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
models_dir = base_dir / 'models'

with open(models_dir/"pca.pkl", "rb") as file:
    pca = pickle.load(file)

with open(models_dir/"class_names.pkl", "rb") as file:
    class_names = pickle.load(file)

with open(models_dir/"best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

num_hidden_layer = best_params['num_hidden_layers']
neuron_per_layer = best_params['neurons_per_layers']
dropout_ratio = best_params['dropout_ratio']
input_dim = pca.n_components_
output_dim = len(class_names)

model = MyNN(input_dim, output_dim, num_hidden_layer, neuron_per_layer, dropout_ratio)

model.load_state_dict(torch.load(models_dir/'ann_model.pth', map_location = 'cpu'))

model.eval()

def predict_csv(file):

    df = pd.read_csv(file, header = None)

    if df.shape[1] != pca.n_features_in_:
        raise ValueError('Input feature size mismatch. Expected 561 features.')

    X_pca = pca.transform(df)

    X_final = torch.tensor(X_pca, dtype = torch.float32)

    with torch.no_grad():

        output = model(X_final)
        probs = torch.softmax(output, dim = 1)
        predictions = output.argmax(dim = 1)
    
    labels = [class_names[p] for p in predictions.numpy()]

    confidence = probs.max(dim = 1).values.numpy()

    return labels, confidence