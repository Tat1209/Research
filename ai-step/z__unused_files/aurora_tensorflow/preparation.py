import tensorflow as tf

# 画像データのサイズ変更と，画素値の正規化を実施する
#   正規化：0～255 → 0.0～1.0
def normalize_img(img, label):
    shape = (85,128,3)
    IMG_HEIGHT = shape[0]   # 画像のサイズ (縦幅) 単位：ピクセル
    IMG_WIDTH = shape[1]    # 画像のサイズ (横幅) 単位：ピクセル

    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])     # 画像のリサイズ 順番入れ替えた
    img = tf.cast(img, tf.float32) / 255.0                  # 画像のデータ型を浮動小数点型に変換する (32bitの浮動小数点)

    return img, label       # 正規化した画素値と，画像の正解ラベルを返す

def preprocess(ds, batch_size):
    # ds = ds.map(lambda img, label: normalize_img(img, label, shape), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def fetch_train_val(train_val_dir, class_names, batch_size, num_train, _):
# def fetch_train_val(train_val_dir, class_names, batch_size, num_train, shape):

# 訓練・検証データを読み込む    read, decode
    ds_train_val = tf.keras.utils.image_dataset_from_directory(train_val_dir, class_names=class_names, seed=0, batch_size=None)

# 訓練データの準備
    ds_train = ds_train_val.take(num_train)
    ds_train = preprocess(ds_train, batch_size)
    # ds_train = preprocess(ds_train, batch_size, shape)

# 検証データの準備
    ds_val = ds_train_val.skip(num_train)
    ds_val = preprocess(ds_val, batch_size)
    # ds_val = preprocess(ds_val, batch_size, shape)
    
    return ds_train, ds_val


# 画像ファイルを読み込んだり，モデルに入力するための形式に変換する処理
def decode_image_from_path(file_path):
    # 画像の読み込み
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img
    # return img, file_path


def fetch_test(test_dir, batch_size, shape):
    # テストデータの読み込みと，モデルに入力するための形式に変換する処理
    ds_test = tf.data.Dataset.list_files(test_dir + "/*.jpg", shuffle=False)
    ds_test = ds_test.map(decode_image_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = preprocess(ds_test, batch_size, shape)
    ds_test = preprocess(ds_test, batch_size)
    
    return ds_test
    
