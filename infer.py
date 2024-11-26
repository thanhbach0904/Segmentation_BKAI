import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F

def preprocess_image(image_path, input_size):

    image = Image.open(image_path).convert('RGB')
    

    transform = transforms.Compose([
        transforms.Resize(input_size), 
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def postprocess_output(output, original_image, input_tensor, output_path):
    output = torch.argmax(output, dim=1).squeeze(0).cpu()  


    one_hot_prediction = F.one_hot(output, num_classes=3).float()  # Assuming 3 classes
    mask_visual = one_hot_prediction.numpy()


    fig, arr = plt.subplots(1, 2, figsize=(12, 6))  

    arr[0].imshow(original_image)
    arr[0].set_title('Original Image')
    arr[0].axis('off')  
    
    arr[1].imshow(mask_visual, interpolation='nearest')  
    arr[1].set_title('Predicted Segmentation')
    arr[1].axis('off')  
    

    plt.tight_layout()
    

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved at {output_path}")
    plt.close(fig) 



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    model.to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'segmentation_efficient_net.pth')  # Update with your checkpoint path
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    input_size = (224, 224)  
    image_tensor, original_image = preprocess_image(args.image_path, input_size)
    image_tensor = image_tensor.to(device)
    
    
    with torch.no_grad():
        output = model(image_tensor) 

   
    output_path = os.path.splitext(args.image_path)[0] + "_visualization.png"

    postprocess_output(output, original_image, image_tensor, output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)
