import os
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

from prep import Prep
from default import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 256             # batchサイズ
num_train = 1080            # ここでは訓練データを 800枚 としているので，残り(400枚) は検証データとなる

epochs = 10                  # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.00001     # 学習率 (重みをどの程度変更するか？)
shape = (85, 128, 3)        # 高さ、幅、RGB

# pr = Prep(data_dir, class_names, batch_size, num_train, shape)
pr = Prep.instance_with_info(batch_size, num_train, shape)

ds_train, ds_val = pr.fetch_train_val()


# モデルの構築
model = model(batch_size, shape, learning_rate, pr.get_classes())       #  この「model」という変数に，構築するモデルのすべての情報が入ります
# plot_model(model, show_shapes=True, expand_nested=True)       # 確認のため，モデルの構造を表示してみます


# 学習を実施します
history = model.fit(ds_train, epochs=epochs, validation_data=ds_val)

# モデルの保存を行います (Google Drive) ファイル名は「competition_model_(日時).h5」になっています
model.save("competition_model_{}.h5".format(datetime.now().strftime("%m%d_%H%M%S")))

ds_test = pr.fetch_test()

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
