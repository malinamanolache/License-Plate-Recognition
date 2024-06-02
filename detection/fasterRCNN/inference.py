import cv2
import os
from torchvision import transforms
import argparse
import torch
import torchvision
import numpy as np


def run_fasterrcnn(model_path: str, input_path: str, out_path: str) -> None:

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=None, 
                                                                num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    if os.path.isfile(input_path):
        img = cv2.imread(input_path)
        img_tensor = transform(img)
        img_tensor = img_tensor.to(device)
        preds = model([img_tensor])

        print(preds)

        color = (0, 0, 255)
        
        for pred in preds:
            box = pred["boxes"].cpu().detach()
            box = np.array(box[0], dtype=np.int64)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        img_path = os.path.join(out_path, "result.png")
        cv2.imwrite(img_path, img)



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory.")
    args=parser.parse_args()

    run_fasterrcnn(args.model_path, args.input, args.out)