from coatnet import coatnet
import tensorflow as tf

def model(shape, learning_rate, class_num):
    model = coatnet.coatnet1(input_shape = shape, include_top = False)

    flatten = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    drop_out = tf.keras.layers.Dropout(0.1)(flatten)
    dense = tf.keras.layers.Dense(2048, activation = "relu")(drop_out)
    prediction = tf.keras.layers.Dense(class_num, activation = "softmax", name = "prediction")(dense)
    model = tf.keras.Model(model.input, prediction)
    loss = tf.keras.losses.sparse_categorical_crossentropy
    opt = tf.keras.optimizers.Adam(learning_rate)
    metric = [tf.keras.metrics.sparse_categorical_accuracy]

    model.compile(loss = loss, optimizer = opt, metrics = metric)
    
    return model