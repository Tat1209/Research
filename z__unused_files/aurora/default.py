import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Softmax

# モデルの構築
#  この「model」という変数に，構築するモデルのすべての情報が入ります
def model(batch_size, shape, learning_rate, class_num):

    model = tf.keras.Sequential()

# モデルの編集 (特徴抽出器)
# 編集場所はここから！
###################################################################################

# 畳み込み - 活性化関数 (ReLU) - プーリング
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPool2D(2))

# 畳み込み - 活性化関数 (ReLU) - プーリング
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPool2D(2))

###################################################################################
# ここまで！

# 分類器 (こちらは編集しない！)
    model.add(Flatten())
    model.add(Dense(class_num))
    model.add(Softmax())


# 最適化関数の設定
#  "パラメータをどのように更新していくか？"という設定項目になります (学習率をこちらで使っています)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 損失関数の設定
#  画像分類に適した設定にしています
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 学習が実施できるように，モデルの設定を完了します
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.build(input_shape=(batch_size, shape[0], shape[1], 3))
    
    return model
