import tensorflow as tf

def dir2data(train_val_dir, class_names, batch_size, num_train, shape):



# 画像のサイズ (横幅) 単位：ピクセル
    IMG_WIDTH = shape[0]
# 画像のサイズ (縦幅) 単位：ピクセル
    IMG_HEIGHT = shape[1]

# 画像データのサイズ変更と，画素値の正規化を実施する
#   正規化：0～255 → 0.0～1.0
    def normalize_img(img, label):
        # 画像のデータ型を浮動小数点型に変換する (32bitの浮動小数点)
        img = tf.cast(img, tf.float32) / 255.0
        # 画像のリサイズ
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        # 正規化した画素値と，画像の正解ラベルを返す
        return img, label


# 訓練・検証データを読み込む
    ds_train_val = tf.keras.utils.image_dataset_from_directory(
        train_val_dir,
        class_names=class_names,
        seed=0,
        batch_size=None
    )

# 訓練データの準備
    ds_train = ds_train_val.take(num_train)
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# 検証データの準備
    ds_val = ds_train_val.skip(num_train)
    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_val