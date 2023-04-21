import tensorflow as tf
from keras.applications import MobileNetV2 as pre_model
# from keras.applications import EfficientNetV2B0 as pre_model

def model(batch_size, shape, learning_rate, class_num):

    base_model = pre_model(
            include_top=False,
            weights="imagenet",
            input_shape=shape,
            )
    
    base_model.trainable = False     # MobileNetV2の重みの凍結
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()         # GlobalAveragePooling (GAP)
    prediction_layer = tf.keras.layers.Dense(class_num)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # 損失関数の設定
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)       # 最適化関数の設定 "パラメータをどのように更新していくか？"という設定項目になります (学習率をこちらで使っています)


    x = inputs = tf.keras.Input(shape)            # 入力
    x = base_model(x, training=False)             # EfficientNetV2
    x = global_average_layer(x)                   # GAP
    x = tf.keras.layers.Dropout(0.2)(x)           # Dropout
    outputs = prediction_layer(x)                 # 全結合層 -> 1

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])     # 学習が実施できるように，モデルの設定を完了します
    model.build(input_shape=(batch_size, shape[0], shape[1], shape[2]))     #  画像分類に適した設定にしています

    return model
