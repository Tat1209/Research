# プログラムで利用する各種パッケージの定義です
import os
# import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# from keras.utils import plot_model

import preparation as pr
import coat as mod

batch_size = 16             # batchサイズ
num_train = 1080            # ここでは訓練データを 800枚 としているので，残り(400枚) は検証データとなる
epochs = 10                 # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.00001       # 学習率 (重みをどの程度変更するか？)
shape = (85, 128, 3)      # 高さ、幅、RGB

# データの読み込み
source_dir = "./aurora/competition01_gray_128x128/"
train_val_dir = source_dir + "train_val"      # 訓練・検証データが格納されているフォルダを指定します
class_names = ["aurora", "clearsky", "cloud", "milkyway"]
test_dir = source_dir + "test"                # テストデータが格納されているフォルダを指定します
ds_train, ds_val = pr.fetch_train_val(train_val_dir, class_names, batch_size, num_train, shape)

# モデルの構築
model = mod.model(batch_size, shape, learning_rate, len(class_names))       #  この「model」という変数に，構築するモデルのすべての情報が入ります
# plot_model(model, show_shapes=True, expand_nested=True)       # 確認のため，モデルの構造を表示してみます


# 学習を実施します
history = model.fit(ds_train, epochs=epochs, validation_data=ds_val)

# モデルの保存を行います (Google Drive) ファイル名は「competition_model_(日時).h5」になっています
model.save("competition_model_{}.h5".format(datetime.now().strftime("%m%d_%H%M%S")))

ds_test = pr.fetch_test(test_dir, batch_size, shape)

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

# # 2,128枚のテストデータがあるので，「0～2127」までで，画像の番号を選択する
# #  img_num = 0 ～ img_num = 2127 までで好きな値を設定してみて下さい
# img_num = 0

# # テスト画像を読み込む
# test_img = preprocess_test_img(test_files[img_num])[0]

# # 画像を出力部分に表示します
# plt.figure()
# plt.imshow(test_img)
# # plt.show()
# plt.savefig("out.jpg")

# # 画像の分類を実施して，結果を表示します
# output = model.predict(np.expand_dims(test_img, axis=0)).argmax()
# print("AIの予測結果 (数値)：{}, 予測結果:{}".format(output, class_names[output]))