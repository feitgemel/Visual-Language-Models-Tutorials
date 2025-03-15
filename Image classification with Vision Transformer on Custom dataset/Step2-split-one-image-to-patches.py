import torch
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from torchvision import transforms
import os
import numpy as np 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchinfo import summary


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print("Torch version:")
print(torch.__version__)

train_dir = 'D:/Data-Sets-Image-Classification/Guava Fruit Disease Dataset/train'
valid_dir = 'D:/Data-Sets-Image-Classification/Guava Fruit Disease Dataset/val'

#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0
print("NUM_WORKERS:", NUM_WORKERS)

def create_dataloaders(
    train_dir: str,
    valid_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader, class_names

IMG_SIZE = 224
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           
print(f"Manually created transforms: {manual_transforms}")

BATCH_SIZE = 32 
#############################################################################

# Devide the image to patches 16X16 

#1- turn an image into patches

#2- flatten the patch feature maps into a single dimension

#3- Convert the output into Desried output (flattened 2D patches): (196, 768) -> N×(P2⋅C) #Current shape: (1, 768, 196)

# 1. Create a class which subclasses nn.Module

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3, # color image
                 patch_size:int=16, # the size of each patch ! 16X16
                 embedding_dim:int=768):  # How many pixels : For each patch 16X16X3 = 768 embedding=pixels
        super().__init__()
        
        # 3. Create a layer to turn an image into patches
        # -> This conv2D help to convert the images to patches !!! like kernels-feature maps 16X16
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size, # We jump each time 16 (no over lapping)
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        # -> create a Flatten layer
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    # -> The result format should be : Number of patchs(=196) X number of pixels (=768)
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    



#Lets create layers used in Transformer's encoder:
#Norm (LN or LayerNorm) - torch.nn.LayerNorm().
#Layer Normalization (torch.nn.LayerNorm() or Norm or LayerNorm or LN) normalizes an input over the last dimension.

#Layer Normalization helps improve training time and model generalization (ability to adapt to unseen data).

#We can implement the MSA layer in PyTorch with torch.nn.MultiheadAttention() with the parameters:
#Multi-Head Self Attention (MSA) - torch.nn.MultiheadAttention()

    #embed_dim - the embedding dimension D .

    #num_heads - how many attention heads to use (this is where the term "multihead" comes from)

    #dropout - whether or not to apply dropout to the attention layer 

    #batch_first - does our batch dimension come first? (yes it does)


# 1. Create a class that inherits from nn.Module

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()
        
        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?
        
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
    


# MLP Block # 
# 1. Create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()
        
        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )
    
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x




# Merge the components 
# Creating a Transformer Encoder by combining our custom made layers

# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
 
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        # Use the MSA block we have created earlier
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # 4. Create MLP block (equation 3)
        # Use the MLP block we have created earlier
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    # 5. Create a forward() method  
    def forward(self, x):
        
        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x 
        
        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x 
        
        return x
    

#Let's build a vision transformer
# 1. Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!
        
        # 3. Make the image size is divisble by the patch size 
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        
        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2
                 
        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        
        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
                
        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
         
        # 8. Create patch embedding layer (Convert image to patches )
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
       
        # 10. Create classifier head (Final layer to classify  )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    # 11. Create a forward() method
    def forward(self, x):
        
        # 12. Get batch size
        batch_size = x.shape[0]
        
        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1) 
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x 








# Run the code :
################


train_dataloader, valid_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    valid_dir=valid_dir,
    transform=manual_transforms, 
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,  # Adjust as needed
)

print(train_dataloader, valid_dataloader, class_names)

# Calculate number of images and batches
num_train_images = len(train_dataloader.dataset)  # Total images in the train dataset
num_train_batches = len(train_dataloader)  # Total batches in the train DataLoader
print(f"Number of training images: {num_train_images}")
print(f"Number of training batches: {num_train_batches}")
print("==============================================")
num_valid_images = len(valid_dataloader.dataset)  # Total images in the validation dataset
num_valid_batches = len(valid_dataloader)  # Total batches in the validation DataLoader
print(f"Number of validation images: {num_valid_images}")
print(f"Number of validation batches: {num_valid_batches}")



# Disaply one Imege from the train 
#  
# Get the first batch of images 
image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]

print("==============================================")
print(image.shape, label)

# Convert the PyTorch tensor to a NumPy array
image_np = image.numpy()

# Rearrange the dimensions from [C, H, W] to [H, W, C]
image_rearranged = np.transpose(image_np, (1, 2, 0))

# Plot the image
plt.title(class_names[label])
plt.imshow(image_rearranged)
plt.axis('off')
plt.show()






# Functions for visualizing patches
# ==========================================

def show_image_with_patches(image_tensor, patch_size=16):
    """
    Displays an image with grid lines showing how it's divided into patches.
    
    Args:
        image_tensor: PyTorch tensor of shape [C, H, W]
        patch_size: Size of patches (default: 16)
    """
    # Convert tensor to numpy
    image_np = image_tensor.numpy()
    
    # Rearrange the dimensions from [C, H, W] to [H, W, C]
    image_rearranged = np.transpose(image_np, (1, 2, 0))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the image
    ax.imshow(image_rearranged)
    
    # Get image dimensions
    h, w = image_rearranged.shape[0], image_rearranged.shape[1]
    
    # Add grid lines to show patches
    # Vertical lines
    for i in range(0, w, patch_size):
        ax.axvline(x=i, color='yellow', linestyle='-', linewidth=1)
    
    # Horizontal lines
    for i in range(0, h, patch_size):
        ax.axhline(y=i, color='yellow', linestyle='-', linewidth=1)
    
    # Add title and remove axis labels
    ax.set_title(f"Image divided into {patch_size}x{patch_size} patches")
    ax.axis('off')
    
    # Calculate total number of patches
    num_patches = (h // patch_size) * (w // patch_size)
    plt.figtext(0.5, 0.01, f"Total patches: {num_patches}", ha="center", fontsize=12, 
                bbox={"facecolor":"yellow", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.show()

def show_image_patches(image_tensor, patch_size=16, num_patches_to_show=4):
    """
    Extracts and displays individual patches from the image.
    
    Args:
        image_tensor: PyTorch tensor of shape [C, H, W]
        patch_size: Size of patches (default: 16)
        num_patches_to_show: Number of random patches to display
    """
    # Convert tensor to numpy
    image_np = image_tensor.numpy()
    
    # Rearrange dimensions
    image_rearranged = np.transpose(image_np, (1, 2, 0))
    
    # Get image dimensions
    h, w = image_rearranged.shape[0], image_rearranged.shape[1]
    
    # Calculate how many patches fit in each dimension
    patches_h = h // patch_size
    patches_w = w // patch_size
    
    # Create a figure to display patches
    fig, axs = plt.subplots(1, num_patches_to_show, figsize=(12, 3))
    
    # Choose random patches to display
    np.random.seed(42)  # For reproducibility
    for i in range(num_patches_to_show):
        # Select random patch coordinates
        patch_h = np.random.randint(0, patches_h)
        patch_w = np.random.randint(0, patches_w)
        
        # Extract the patch
        patch = image_rearranged[
            patch_h * patch_size:(patch_h + 1) * patch_size,
            patch_w * patch_size:(patch_w + 1) * patch_size,
            :
        ]
        
        # Display the patch
        axs[i].imshow(patch)
        axs[i].set_title(f"Patch ({patch_h},{patch_w})")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Random {patch_size}x{patch_size} patches from the image", y=1.05)
    plt.show()









# Visualize the image divided into patches
print("\nVisualizing the image divided into patches:")
patch_size = 16
show_image_with_patches(image, patch_size=patch_size)

# Show some individual patches
print("\nDisplaying some individual patches:")
show_image_patches(image, patch_size=patch_size, num_patches_to_show=4)

##########################################################################
# PatchEmbedding layer ready
# Let's test it on single image !!!! 
#######################################

# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


# set permambebt seed for random 
set_seeds()

# Create an instance of patch embedding layer based on the class
patchify = PatchEmbedding(in_channels=3,
                        patch_size=16,
                        embedding_dim=768)

# Check a single image and get the converted image to flatten image applying the patches 
print(f"Input image shape: {image.unsqueeze(0).shape}")
patch_embedded_image = patchify(image.unsqueeze(0)) # add an extra batch dimension on the 0th index, otherwise will error
print(f"Output patch embedding shape: {patch_embedded_image.shape}")

# The result is [1 , 196 , 768]
# -> 1  means batch 
# -> 196 means that this image was split to 196 patches
# 768 -> is the number of pixes / embeding / pixesls in each patch 
##################################################

# Visualize the patch embeddings
#print("\nVisualizing patch embeddings:")
#visualize_patch_embeddings(patch_embedded_image)

# lets see the 768 values :

# lets see the values for each patch
# 
print(patch_embedded_image) 
print(f"Patch embedding shape: {patch_embedded_image.shape} -> [batch_size, number_of_patches, embedding_dimension]")