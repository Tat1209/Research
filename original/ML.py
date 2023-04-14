import glob
# プログラムで利用する各種パッケージの定義です
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Softmax

# 訓練・検証データが格納されているフォルダを指定します
train_val_dir = "aurora/competition01_gray_128x128/train_val"

# テストデータが格納されているフォルダを指定します
test_dir = "aurora/competition01_gray_128x128/test"

# 画像のサイズ (横幅) 単位：ピクセル
IMG_WIDTH = 128
# 画像のサイズ (縦幅) 単位：ピクセル
IMG_HEIGHT = 85

# 今回は4種類に分類を行います (classesには「4」が入る)
#   aurora, clearsky, cloud, milkyway
classes = 4

# モデルが出力する分類結果 (数値)と，名前を紐づける
#   aurora: 0, clearsky: 1, cloud: 2, milkyway: 3
class_names = ["aurora", "clearsky", "cloud", "milkyway"]

###################################################################################
# こちらの項目は「ハイパーパラメータ」と呼ばれる項目になります
# 値を変更してモデルの精度を向上させてみましょう！
###################################################################################
# バッチサイズ (並列して学習を実施する数)
batch_size = 256

# エポック数 (学習を何回実施するか？という変数)
epochs = 5

# 学習率 (重みをどの程度変更するか？)
learning_rate = 0.001
               
# 訓練・検証データ画像が 1,200 枚あるので，これを訓練データと検証データに分割する
#  ここでは訓練データを 800枚 としているので，残り(400枚) は検証データとなる
num_train = 800


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

# モデルの構築
#  この「model」という変数に，構築するモデルのすべての情報が入ります
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
model.add(Dense(classes))
model.add(Softmax())


# 最適化関数の設定
#  "パラメータをどのように更新していくか？"という設定項目になります (学習率をこちらで使っています)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 損失関数の設定
#  画像分類に適した設定にしています
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 学習が実施できるように，モデルの設定を完了します
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.build(input_shape=(batch_size, IMG_HEIGHT, IMG_WIDTH, 3))

# 確認のため，モデルの構造を表示してみます
# plot_model(model, show_shapes=True, expand_nested=True)


# 学習を実施します
history = model.fit(ds_train,
                    epochs=epochs,
                    validation_data=ds_val)

# モデルの保存を行います (Google Drive)
#   Google Driveにある「aistep_output」というフォルダの中に保存されます
#   ファイル名は「competition_model_(日時).h5」になっています
model.save("competition_model_{}.h5".format(datetime.now().strftime("%m%d_%H%M%S")))
           
           
# 画像ファイルを読み込んだり，モデルに入力するための形式に変換する処理
def preprocess_test_img(file_path):
    # 画像の読み込み
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    # 解像度を変更し，値の正規化を実施する
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img, file_path


# テストデータの読み込みと，モデルに入力するための形式に変換する処理
ds_test = tf.data.Dataset.list_files(test_dir + "/*.jpg", shuffle=False)
ds_test = ds_test.map(preprocess_test_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# モデルによって分類を実施し，画像が何か？ということを予測する
result = model.predict(ds_test)

###################################################################################
# すべてのテストデータの予測が完了したら，結果のファイルを「CSV」形式で出力します
#  こちらのファイルを提出して下さい！！
###################################################################################
# ファイル名一覧を取得する
test_files = []
for item in iter(ds_test):
    filenames = item[1]
    for file in filenames.numpy():
        test_files.append(str(file)[2:-1])

# 「competition_result_(現在日時).csv」というファイル名で保存されます
with open('competition_result_{}.csv'.format(datetime.now().strftime("%m%d_%H%M%S")), 'w', newline='') as f:
    writer = csv.writer(f)
    # テストデータ1枚に対して，結果を1行ずつ出力していきます
    for i, item in enumerate(result):
        # ファイル名
        writer.writerow([
            Path(test_files[i]).name,  # 画像のファイル名
            np.argmax(item)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
        ])
        
# 2,128枚のテストデータがあるので，「0～2127」までで，画像の番号を選択する
#  img_num = 0 ～ img_num = 2127 までで好きな値を設定してみて下さい
img_num = 0

# テスト画像を読み込む
test_img = preprocess_test_img(test_files[img_num])[0]

# 画像を出力部分に表示します
plt.figure()
plt.imshow(test_img)
# plt.show()
plt.savefig("out.jpg")

# 画像の分類を実施して，結果を表示します
output = model.predict(np.expand_dims(test_img, axis=0)).argmax()
print("AIの予測結果 (数値)：{}, 予測結果:{}".format(output, class_names[output]))