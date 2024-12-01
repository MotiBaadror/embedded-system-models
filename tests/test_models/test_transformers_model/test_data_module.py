import os
from dataclasses import dataclass

import datasets
import pandas as pd
from _pytest.fixtures import fixture
from datasets import Dataset

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule



class TestDataModule:
    @fixture
    def result(self):
        @dataclass
        class Result:
            dm = GLUEDataModule(
                "distilbert-base-uncased"
            )
            # dm.prepare_data()
            # dm.setup()
            #

        return Result()

    def test_init(self,result):
        assert isinstance(result.dm,GLUEDataModule)


    def test_prepare_data(self,result):
        result.dm.prepare_data()


    def test_setup(self,result):
        # result.dm.setup()
        print(result.dm.dataset)

    def test_train_dataloader(self,result):
        for data in result.dm.train_dataloader():
            print(data)
            break


def test_data_loaded():
    file_path = add_rootpath('data/raw_data/version_0/spam_ham_dataset.csv')
    df = pd.read_csv(file_path)
    df = df[['text','label_num']]
    df.columns = ['text','labels']
    print(df.shape)
    d=dict(
        text=df.iloc[:,0].values.tolist(),
        labels=df.iloc[:,1].values.tolist()
    )
    print(d['text'][1])
    print(d['labels'][0])
    dataset = datasets.Dataset.from_dict(d)
    splits = dataset.train_test_split(test_size=0.4)
    dataset_dict = datasets.DatasetDict({
        "train": splits['train'],
        "test": splits['test']
    })

    os.makedirs('./data/my_splits',exist_ok=True)
    # dataset.save_to_disk('./data/my_splits')
    dataset_dict.save_to_disk('./data/my_splits')
    print(dataset)
    # dataset_dict = df.to_dict()
    # for i in dataset_dict:
    #     print(dataset_dict[i])
    #     break
    # print(dataset_dict)

    # dataset = datasets.load_dataset('csv',data_files= )
    # print(dataset)