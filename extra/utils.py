import torch


def get_layer_sizes_with_hooks(model_path, input_data):
    """
    Loads a PyTorch model, registers hooks, performs a forward pass to capture output shapes.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # print(model['model_sd'])
    for key, value in model['model_sd'].items():
        print(key, value.shape)

    return 1
    
# Example usage
model_path = 'D:/tudelft/CS4240-Reproducibility-Project/save/models/epoch-20.pth'
layer_sizes = get_layer_sizes_with_hooks(model_path, [(4, 5, 1, 3, 80, 80), (4, 75, 3, 80, 80)])
print(f"Layer sizes: {layer_sizes}")