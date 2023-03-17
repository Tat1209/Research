import tensorflow as tf
# GPUが利用可能か確認する
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# TensorFlowがGPUを使用しているか確認する
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())
