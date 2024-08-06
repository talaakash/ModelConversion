import coremltools
import coremltools as ct
import torch
from coremltools.models.neural_network import quantization_utils

from iopaint.model.lama import LaMa

class CoreMLaMa(torch.nn.Module):
    def __init__(self, lama):
        super(CoreMLaMa, self).__init__()
        self.lama = lama

    def forward(self, image, mask):
        normalized_mask = ((mask > 0) * 1).byte()
        lama_out = self.lama(image, normalized_mask)
        output = torch.clamp(lama_out * 255, min=0, max=255)
        return output

model_manager = LaMa("cpu")

# Fixed image/mask size
# Flexible input shapes are not (currently) supported, for various reasons
size = (800, 800) # pixel width x height

# Image/mask shapes in PyTorch format
image_shape=(1, 3, size[1], size[0])
mask_shape=(1, 1, size[1], size[0])

lama_inpaint_model = model_manager.model
model = CoreMLaMa(lama_inpaint_model).eval()

print("Scripting CoreMLaMa")
jit_model = torch.jit.script(model)

print("Converting model")
# Note that ct.ImageType assumes an 8 bpp image, while LaMa
# uses 32-bit FP math internally. Creating a CoreML model
# that can work with 32-bit FP image inputs is on the "To Do"
# list
weights_dir = 'Data/com.apple.CoreML/weights'
coreml_model = ct.convert(
    jit_model,
    convert_to=None,
    compute_precision=8,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    inputs=[
        ct.ImageType(name="image",
                     shape=image_shape,
                     scale=1/255.0),
        ct.ImageType(
            name="mask",
            shape=mask_shape,
            color_layout=ct.colorlayout.GRAYSCALE)
    ],
    outputs=[ct.ImageType(name="output")],
    # skip_model_load=True,
)
print("Model converted")

# model_fp16 = quantization_utils.quantize_weights(coreml_model, nbits=16, weights_dir=weights_dir)
# model_fp16.save("quantized_model.mlmodel")

coreml_model_file_name = "LaMaTest.mlpackage"
print(f"Saving model to {coreml_model_file_name}")
coreml_model.save(coreml_model_file_name)
print("Done!")