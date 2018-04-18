import pandas as pd
import sys
from alpenglow.experiments import ExternalModelExperiment

mode = "write"
if(len(sys.argv)>1 and sys.argv[1] == "read"):
    mode = "read"

data = pd.read_csv('../tutorial_data/data.csv', header=None, names=['time', 'user', 'item'])

exp = ExternalModelExperiment(
    period_length=60 * 60 * 24 * 7,
    out_name_base="batches/batch",
    mode=mode
)
res = exp.run(data)
