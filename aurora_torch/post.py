import csv
from pathlib import Path
from datetime import datetime


def postprocess(dl_test, result, model):
# ファイル名一覧を取得する
    test_files = []
    for item in iter(dl_test):
        filenames = item[1]
        for file in filenames: test_files.append(str(file))
            
# # 「competition_result_(現在日時).csv」というファイル名で保存されます
    with open('competition_result_{}.csv'.format(datetime.now().strftime("%m%d_%H%M%S")), 'w', newline='') as f:
        writer = csv.writer(f)
        # テストデータ1枚に対して，結果を1行ずつ出力していきます
        for i, item in enumerate(result):
        # for i, item in enumerate(result):
            writer.writerow([
                Path(test_files[i]).name, item # 画像のファイル名
                # np.argmax(item)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
            ])

    model.save("competition_model_{}.pth".format(datetime.now().strftime("%m%d_%H%M%S")))
            
# 2,128枚のテストデータがあるので，「0～2127」までで，画像の番号を選択する
#  img_num = 0 ～ img_num = 2127 までで好きな値を設定してみて下さい
# img_num = 0

# テスト画像を読み込む
# test_img = preprocess_test_img(test_files[img_num])[0]

# # 画像を出力部分に表示します
# plt.figure()
# plt.imshow(test_img)
# # plt.show()
# plt.savefig("out.jpg")

# 画像の分類を実施して，結果を表示します
# output = model.predict(np.expand_dims(test_img, axis=0)).argmax()
# print("AIの予測結果 (数値)：{}, 予測結果:{}".format(output, class_names[output]))


