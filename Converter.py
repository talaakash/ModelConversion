import coremltools as ct
import tensorflow as tf
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    # Load the Core ML model
    print("Model loading started....")
    mlmodel = ct.models.MLModel('RMBG.mlmodel')
    print("Model is loaded")

    # Convert the Core ML model to a TensorFlow SavedModel
    print("Model conversion to TensorFlow started...")
    tf_model = ct.convert(mlmodel, source='tensorflow')
    print("Model converted to TensorFlow")

    # Save the TensorFlow SavedModel to a directory
    tf.saved_model.save(tf_model, 'saved_model')
    print("TensorFlow SavedModel saved")

    # Convert the SavedModel to a .tflite file
    print("TFLite conversion started...")
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
    tflite_model = converter.convert()
    print("TFLite conversion finished")

    # Save the .tflite file
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'model.tflite'")

except Exception as e:
    print(f"An error occurred: {e}")