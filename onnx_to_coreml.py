import coremltools as ct

model = ct.converters.onnx.convert(
    model='model.onnx',
    mode='classifier',
    minimum_ios_deployment_target='13'
)
model.save('model.mlmodel')