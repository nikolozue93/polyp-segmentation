import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)

model.eval()

# One artificial RGB image, just to test the architecture.
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    output = model(x)

print("Input:", x.shape)
print("Output:", output.shape)

assert output.shape == (1, 1, 256, 256)