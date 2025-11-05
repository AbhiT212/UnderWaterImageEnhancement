import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ConvBlock(nn.Module):
    """Standard Convolutional Block with optional Channel Attention"""
    def __init__(self, in_channels, out_channels, use_ca=False):
        super(ConvBlock, self).__init__()
        self.use_ca = use_ca
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if self.use_ca:
            self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ca:
            ca_weights = self.ca(x)
            x = x * ca_weights
        return x

class DeconvBlock(nn.Module):
    """Deconvolutional Block"""
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.deconv(x)

class P2PNet(nn.Module):
    """
    P2PNet for Underwater Image Enhancement.
    The architecture maintains constant spatial resolution (HxW) throughout.
    """
    def __init__(self, base_ch=32):
        super(P2PNet, self).__init__()

        # --- Encoder Path ---
        # Note: All convs use padding=1 to keep HxW constant
        self.enc1 = ConvBlock(3, base_ch, use_ca=True)         # Fin -> F2in_CA
        self.enc2 = ConvBlock(base_ch, base_ch * 2, use_ca=True) # F2in -> F3in_CA
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 2, use_ca=True) # F3in -> F3out_CA

        # --- Decoder Path ---
        # Note: Since spatial res doesn't change, "Deconv" here is just a regular conv
        # to process features. We use Conv2d instead of ConvTranspose2d.

        # Deconv1 -> F4out
        self.dec1_conv = ConvBlock(base_ch * 2, base_ch * 2) 

        # Concat(F4out, F3in_CA) -> Deconv2 -> F5out
        self.dec2_conv = ConvBlock(base_ch * 4, base_ch) 

        # Concat(F5out, F2in_CA) -> Deconv3 -> F6out
        self.dec3_conv = ConvBlock(base_ch * 2, base_ch) 

        # --- Final Layer ---
        # Concat(F6out, Fin) -> Final Conv -> Sigmoid -> Y
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch + 3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is the input Fin

        # Encoder
        enc1_out = self.enc1(x)       # F2in_CA
        enc2_out = self.enc2(enc1_out) # F3in_CA
        enc3_out = self.enc3(enc2_out) # F3out_CA -> This corresponds to the bottleneck feature

        # Decoder
        dec1_out = self.dec1_conv(enc3_out) # F4out

        # Skip connection from enc2_out (F3in_CA)
        dec2_in = torch.cat([dec1_out, enc2_out], dim=1)
        dec2_out = self.dec2_conv(dec2_in) # F5out

        # Skip connection from enc1_out (F2in_CA)
        dec3_in = torch.cat([dec2_out, enc1_out], dim=1)
        dec3_out = self.dec3_conv(dec3_in) # F6out

        # Final connection with input image Fin
        final_in = torch.cat([dec3_out, x], dim=1)
        output = self.final_conv(final_in) # Y

        return output

if __name__ == '__main__':
    # Test forward pass
    model = P2PNet(base_ch=32)
    dummy_input = torch.randn(8, 3, 256, 256) # (B, C, H, W)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert dummy_input.shape == output.shape
    print("Forward pass successful!")