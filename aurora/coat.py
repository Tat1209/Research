import tensorflow as tf
import coatnet

def model(batch_size, shape, learning_rate, class_num):
    tf.keras.initializers.RandomNormal()
    model = coatnet.coatnet0(input_shape = shape, include_top = False)

    flatten = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    drop_out = tf.keras.layers.Dropout(0.25)(flatten)
    dense = tf.keras.layers.Dense(2048, activation = "relu")(drop_out)
    prediction = tf.keras.layers.Dense(class_num, activation = "softmax", name = "prediction")(dense)

    model = tf.keras.Model(model.input, prediction)

    loss = tf.keras.losses.sparse_categorical_crossentropy
    opt = tf.keras.optimizers.Adam(learning_rate)
    metric = [tf.keras.metrics.sparse_categorical_accuracy]

    model.compile(loss = loss, optimizer = opt, metrics = metric)
    # model.build(input_shape=(batch_size, shape[0], shape[1], 3))
    
    return model