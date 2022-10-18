import glob
import json

import pandas as pd

file_regex = "./results2/*"
data = []
paths = []
for path in glob.glob(file_regex + ".json"):
    paths.append(path)
    with open(path) as json_file:
        data.append(json.load(json_file))

df = pd.DataFrame.from_records(data)

