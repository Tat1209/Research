import csv
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd


def postprocess(pr, result, hist, model):
    test_files = []
    dl_test = pr.fetch_test()
    for item in iter(dl_test):
        filenames = item[1]
        for file in filenames: test_files.append(str(file))
            
    ft = datetime.now().strftime("%m%d_%H%M%S")
    with open(f'competition_result_{ft}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, res in enumerate(result): writer.writerow([Path(test_files[i]).name, res])
    
    pd.DataFrame(hist).to_csv(f'competition_hist_{ft}.csv', index=False)

    save_path = f"competition_model_{ft}.pth"
    torch.save(model, save_path)
            


