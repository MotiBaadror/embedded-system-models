import pandas as pd

from dir_configs import add_rootpath

data = pd.read_csv(add_rootpath('data/raw_data/version_0/spam_ham_dataset.csv'))

print(data.head())
