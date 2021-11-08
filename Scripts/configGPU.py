import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    
    # Returns
        A list of available GPU devices.
    """
    print("tf.__version__ is", tf.__version__)
    print("tf.keras.__version__ is:", tf.keras.__version__)
    print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
        
        
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


def gpu_mem():
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
