import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
def load_model(checkpoint_path, device):
    # Load the trained model
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image_path, input_size):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize to model's expected input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def postprocess_output(output, original_image):
    # Ensure output is a PyTorch tensor (it should be if coming directly from the model)
    if isinstance(output, torch.Tensor):
        output = torch.argmax(output, dim=1).squeeze(0)  # Get class predictions and remove batch dimension
    else:
        raise TypeError(f"Expected output to be a tensor, but got {type(output)}")

    # Move output to CPU for further processing (in case it's on GPU)
    output = output.cpu().numpy()  # Convert to NumPy for visualization

    # Create the mask image
    mask = (output * 255).astype('uint8')  # Scale for visualization
    mask_image = Image.fromarray(mask).convert('RGBA')  # Convert to RGBA for overlay

    # Resize the mask to match the original image size
    mask_image = mask_image.resize(original_image.size)

    # Create an overlay of the mask on the original image
    overlay_image = Image.blend(
        original_image.convert('RGBA'),  # Convert original image to RGBA
        mask_image, 
        alpha=0.5  # Adjust transparency for overlay effect
    )
    
    # Combine original image and overlay side-by-side
    combined_image = Image.new('RGB', (original_image.width * 2, original_image.height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(overlay_image.convert('RGB'), (original_image.width, 0))
    
    return combined_image

def plot_images(output_image):
    plt.title("Segmented Output")
    plt.imshow(output_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    model.to(device)
    # Load the model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model.pth')  # Update with your checkpoint path
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    
    # Preprocess the input image
    input_size = (224, 224)  # Adjust as per your model's input requirement
    image_tensor, original_image = preprocess_image(args.image_path, input_size)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Postprocess the output
    segmented_image = postprocess_output(output, original_image)
    plot_images(segmented_image)
    # Save the result
    #output_path = os.path.splitext(args.image_path)[0] + "_segmented.png"
    #segmented_image.save(output_path)
    #print(f"Segmented image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)



#python infer.py --image_path "C:\Users\dmin\HUST\20241\DeepLearning\Segmentation\bkai-igh-neopolyp\test\test\0a5f3601ad4f13ccf1f4b331a412fc44.jpeg"
