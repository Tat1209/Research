import csv
from pathlib import Path
from datetime import datetime

import torch


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
        for i, res in enumerate(result): writer.writerow([Path(test_files[i]).name, res])

    save_path = "competition_model_{}.pth".format(datetime.now().strftime("%m%d_%H%M%S"))
    torch.save(model, save_path)
            


