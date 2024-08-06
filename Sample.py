import coremltools
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# Load the .mlmodel file
print("Starting...")
model_fp32 = ct.models.MLModel('RMBG.mlmodel')
print("Loaded")
# Quantize the model to 16-bit
model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
print("Quantization done")
# Save the quantized model
model_fp16.save('RMBG_quantized.mlmodel')
print("Quantized model saved as 'RMBG_quantized.mlmodel'")