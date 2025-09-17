import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class TrainableGaussianBlurDownsample(nn.Module):
    """
    Gaussian blur and downsample
    """
    def __init__(self, in_channels, kernel_size=5, sigma=1.0):
        super(TrainableGaussianBlurDownsample, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,  # Downsample by factor of 2
            padding=self.padding,
            groups=in_channels,  # Depthwise convolution (separate kernel per channel)
            bias=False
        )

        self._initialize_gaussian_weights(sigma)
    
    def _initialize_gaussian_weights(self, sigma):
        """Initialize the convolution weights with Gaussian kernel"""
        with torch.no_grad():
            # Create Gaussian kernel
            x = torch.arange(-self.kernel_size//2+1, self.kernel_size//2+1, dtype=torch.float32)
            y = torch.arange(-self.kernel_size//2+1, self.kernel_size//2+1, dtype=torch.float32)
            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
            
            # Calculate Gaussian kernel
            kernel = torch.exp(-0.5 * (x_grid**2 + y_grid**2) / sigma**2)
            kernel = kernel / kernel.sum()  # Normalize
            
            # Set weights for each channel (depthwise convolution)
            for i in range(self.conv.weight.shape[0]):
                self.conv.weight[i, 0] = kernel
    
    def forward(self, x):
        return self.conv(x)


class TrainableUpsample(nn.Module):
    """
    Trainable upsample using ConvTranspose2d (deconvolution)
    """
    def __init__(self, in_channels, kernel_size=4, stride=2):
        super(TrainableUpsample, self).__init__()
        self.padding = (kernel_size - stride) // 2
        
        # Transposed convolution for upsampling
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=in_channels,  # Depthwise transposed convolution
            bias=False
        )
        
        # Initialize with bilinear-like weights
        self._initialize_bilinear_weights()
    
    def _initialize_bilinear_weights(self):
        """Initialize with bilinear interpolation-like weights"""
        with torch.no_grad():
            kernel_size = self.conv_transpose.kernel_size[0]
            # Create bilinear kernel
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            
            og = torch.arange(kernel_size, dtype=torch.float32)
            filt = (1 - torch.abs(og - center) / factor)
            filt = torch.outer(filt, filt)
            
            # Set weights for each channel
            for i in range(self.conv_transpose.weight.shape[0]):
                self.conv_transpose.weight[i, 0] = filt
    
    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class TrainableGaussianPyramid(nn.Module):
    """
    Trainable Gaussian pyramid using learnable convolutions
    """
    def __init__(self, in_channels, levels=5, kernel_size=5, sigma=1.0):
        super(TrainableGaussianPyramid, self).__init__()
        self.levels = levels
        
        # Create trainable blur+downsample layers for each level
        self.blur_downsample_layers = nn.ModuleList([
            TrainableGaussianBlurDownsample(in_channels, kernel_size, sigma)
            for _ in range(levels - 1)
        ])
    
    def forward(self, image):
        """
        Build a Gaussian pyramid from the input image with batch support.
        Args:
            image (Tensor): Input image of shape (B, C, H, W).
        Returns:
            List[Tensor]: List of Gaussian pyramids, each tensor of shape (B, C, H', W').
        """
        pyramid = [image]
        current_image = image
        
        for blur_downsample in self.blur_downsample_layers:
            current_image = blur_downsample(current_image)
            pyramid.append(current_image)
        
        return pyramid

class TrainableLaplacianPyramid(nn.Module):
    """
    Trainable Laplacian pyramid using learnable upsampling
    """
    def __init__(self, in_channels, levels=5, upsample_kernel_size=4):
        super(TrainableLaplacianPyramid, self).__init__()
        self.levels = levels
        
        # Create trainable upsample layers
        self.upsample_layers = nn.ModuleList([
            TrainableUpsample(in_channels, upsample_kernel_size)
            for _ in range(levels - 1)
        ])
    
    def forward(self, gaussian_pyramid):
        """
        Build a Laplacian pyramid from a Gaussian pyramid with batch support.
        Args:
            gaussian_pyramid (List[Tensor]): List of tensors representing the Gaussian pyramid.
        Returns:
            List[Tensor]: Laplacian pyramid of images.
        """
        laplacian_pyramid = []
        
        for i in range(len(gaussian_pyramid) - 1):
            # Upsample the next level to match current level's size
            target_size = gaussian_pyramid[i].shape[2:]
            upsampled = self.upsample_layers[i](gaussian_pyramid[i + 1])
            laplacian_pyramid.append(gaussian_pyramid[i] - upsampled)
        
        # Last level is just the smallest image
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

class TrainablePyramidNet(nn.Module):
    """
    Complete trainable pyramid network combining Gaussian and Laplacian pyramids
    """
    def __init__(self, in_channels, levels=5, gaussian_kernel_size=5, sigma=1.0, upsample_kernel_size=4):
        super(TrainablePyramidNet, self).__init__()
        self.gaussian_pyramid = TrainableGaussianPyramid(
            in_channels, levels, gaussian_kernel_size, sigma
        )
        self.laplacian_pyramid = TrainableLaplacianPyramid(
            in_channels, levels, upsample_kernel_size
        )
    
    def forward(self, image):
        """
        Args:
            image (Tensor): Input image of shape (B, C, H, W).
        Returns:
            Tuple[List[Tensor], List[Tensor]]: (Gaussian pyramid, Laplacian pyramid)
        """
        gaussian_pyr = self.gaussian_pyramid(image)
        laplacian_pyr = self.laplacian_pyramid(gaussian_pyr)
        return gaussian_pyr, laplacian_pyr


class IlluminationPrompter(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, 
                 levels=3, gaussian_kernel_size=5, sigma=1.0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.levels = levels
        self.gaussian_kernel_size = gaussian_kernel_size
        self.sigma = sigma
        self.embed_dim_per_level = embed_dim // levels

        self.pyramid = TrainablePyramidNet(in_chans, levels, gaussian_kernel_size, self.sigma)
    
        # projection layers to process laplacian
        self.proj = nn.ModuleList()  # do not use list, or the modules in the python list cannot use .to(device)
        for i in range(self.levels):
            kernel_size_per_level = tuple([item // (2 ** i) for item in self.patch_size])
            if i != self.levels - 1:
                self.proj.append(nn.Conv2d(in_channels = 3, 
                                           out_channels = self.embed_dim_per_level, 
                                           kernel_size = kernel_size_per_level, 
                                           stride = kernel_size_per_level[0]))
            else:
                self.proj.append(nn.Conv2d(in_channels = 3, 
                                           out_channels = self.embed_dim - self.embed_dim_per_level * (self.levels - 1), 
                                           kernel_size = kernel_size_per_level, 
                                           stride = kernel_size_per_level[0]))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x, return_gaussian=False, return_laplacian=False):
        gaussiam_x, lapulacian_x = self.pyramid(x)
        proj_x = []
        for idx, _x in enumerate(lapulacian_x):
            proj_x.append(self.proj[idx](_x))
        x = torch.cat(proj_x, dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.embed_dim)
        x = self.norm(x)
        if return_gaussian and return_laplacian:
            return x, gaussiam_x, lapulacian_x
        elif return_gaussian:
            return x, gaussiam_x
        elif return_laplacian:
            return x, lapulacian_x
        else:
            return x
