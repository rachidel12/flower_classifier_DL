import numpy as np
from torchvision import models
import torch
import argparse
import json
from PIL import Image
from prettytable import PrettyTable

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img= Image.open(img_path)
    img= img.resize((256,256))
    w, h = img.size
    left = (w - 224) / 2
    right = (w + 224) / 2
    top = (h - 224) / 2
    bottom = (h + 224) / 2
    img = img.crop((left, top, right, bottom))
    img= np.array(img)/255
    means=np.array([0.485, 0.456, 0.406])
    sd= np.array([0.229, 0.224, 0.225])
    img= (img-means)/sd
    img=img.transpose(2,0,1)
    return img


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    
    arch = checkpoint["arch"]
    my_local = dict()
    exec("model = models.{}(pretrained=True)".format(arch), globals(), my_local)

    model =  my_local['model']
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    class_idx_mapping = checkpoint["class_idx_mapping"]
    idx_class_mapping = {j: i for i, j in class_idx_mapping.items()}
    
    return model, idx_class_mapping


def predict(image_path, checkpoint_file, topk=5, device="cpu"):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model, idx_class_mapping = load_model(checkpoint_file)
    model.to(device)
    model.eval()
     
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        outputs = model.forward(img_tensor)
    
    ps = torch.exp(outputs)
    probs, indices = ps.topk(topk)
    
    probs, indices = probs.to('cpu'), indices.to('cpu')
    probs = probs.numpy().squeeze()
    indices = indices.numpy().squeeze()
    classes = [idx_class_mapping[i] for i in indices]
    
    return probs, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image-path", help="Path to the image to be classified.")

    ap.add_argument("model-checkpoint", help="Checkpoint path to load the model from. ")
    
    ap.add_argument("--top_k", help="Specify number of K most likely classes to display", default=5, type=int)

    ap.add_argument("--category_names", help="Mapping of categories to real names. (default: cat_to_name.json)", 
                    default="cat_to_name.json")

    ap.add_argument("--gpu", help="Use GPU or CPU for training", action="store_true")
    args = vars(ap.parse_args())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(args)

    if args["gpu"] and device == "cuda":
        device = "cuda"
    elif args["gpu"] and device == "cpu":
        print("CUDA not found on device, using CPU instead!")
        device = "cpu"
    else:
        device = "cpu"

    probs, classes = predict(image_path=args["image-path"], checkpoint_file=args["model-checkpoint"], 
                            topk=args["top_k"], device=device)

    with open(args["category_names"], 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[c] for c in classes]


    x = PrettyTable()
    x.field_names = ["Class Name", "Probability"]
    for c,p in zip(class_names, probs):
        x.add_row([c, p])
    
    print(x)

if __name__ == '__main__':
    main()