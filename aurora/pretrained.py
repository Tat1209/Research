from keras.applications import EfficientNetV2B0 as model_app

def model():
    base_model = model_app(
            include_top=False,
            weights="imagenet",
            input_shape=None,
            )

    return base_model

# def default_model():
# # モデルの編集 (特徴抽出器)

# # 畳み込み - 活性化関数 (ReLU) - プーリング
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
#     model.add(ReLU())
#     model.add(MaxPool2D(2))

# # 畳み込み - 活性化関数 (ReLU) - プーリング
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(ReLU())
#     model.add(MaxPool2D(2))
