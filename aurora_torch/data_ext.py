import os
import fnmatch
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

folder = "/root/app/"
pattern = "*result*"

paths = []
for filename in os.listdir(folder): 
    if fnmatch.fnmatch(filename, pattern): paths.append(os.path.join(folder, filename))
paths = sorted(paths)
paths = paths[-20:]
print(len(paths))

dfs = [pd.read_csv(path, header=None, names=["path", "tensor"]) for path in paths]

# 空のDataFrameを作る
# リストの要素ごとに処理する
df_classed = pd.DataFrame(columns=["path", "0", "1", "2", "3"])
for i, df in enumerate(dfs):
    df_tmp = pd.DataFrame(columns=["path", "0", "1", "2", "3"])
    for s in df["tensor"]:
        # 文字列をリストに変換する
        s = s.strip("[]").split()
        # リストの要素を浮動小数点に変換する
        s = [float(x) for x in s]
        # リストをDataFrameに変換して、元のDataFrameに追加する
        if i == 0: df_classed = df_classed._append(pd.DataFrame([s], columns=['0', '1', '2', '3']), ignore_index=True)[['0', '1', '2', '3']]
        else: df_tmp = df_tmp._append(pd.DataFrame([s], columns=['0', '1', '2', '3']), ignore_index=True)[['0', '1', '2', '3']]
    if i != 0: df_classed += df_tmp
df_classed /= len(dfs)
df_classed["class"] = df_classed.idxmax(axis=1)
df_classed["path"] = dfs[0]["path"]

df_out = df_classed[["path", "class"]]

ft = datetime.now().strftime("%m%d_%H%M%S")
pd.DataFrame(df_out).to_csv(f'competition_result_{ft}.csv', index=False, header=False)
    


