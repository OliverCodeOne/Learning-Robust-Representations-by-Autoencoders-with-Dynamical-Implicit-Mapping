import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop("classify", axis=1).values
        self.labels = dataframe["classify"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = torch.Tensor(self.features[index])
        label = self.labels[index]  # 将标签转换为张量

        return feature, label

def main():
    # 读取
    df = pd.read_excel(r'D:\Rookie Project\Pycharm Project\TAVIdata\surgery_data\TAVI_new\论文\dataset\dataset - 平衡.xlsx')
    dataset = DataFrameDataset(df)

    # 定义批量大小（batch size）和是否进行随机洗牌（shuffle）
    batch_size = 32
    shuffle = True

    # 创建DataLoader对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 使用DataLoader迭代数据
    for batch_id,sample in enumerate(dataloader):
        # 在这里对每个batch的数据进行处理
        # batch是一个tensor，可以直接传递给模型进行训练
        print(sample[1])
if __name__ == '__main__':
    main()