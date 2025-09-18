import json
import os
import time

import requests
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("form-directories.csv", dtype=str)
selected = df.groupby(["cik", "form"]).last().sort_values("date", ascending=False).reset_index()

for index, row in tqdm(selected.iterrows(), total=len(selected)):
    filename = f"form-directories/{row['cik']}.json"

    response = requests.get(row["directory"], headers={"User-Agent": "Jim Pivarski jpivarski@uchicago.edu"})
    test_json = json.loads(response.content)
    assert "directory" in test_json

    with open(filename, "w") as file:
        json.dump(test_json, file, separators=(",", ":"))

    time.sleep(0.1)
