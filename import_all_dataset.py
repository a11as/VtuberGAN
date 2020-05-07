from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# データセット読み込み
# dataset配下のサブフォルダー内の画像をtensor型に変換
dataset = datasets.ImageFolder("./dataset/",
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
### 確認用
# print(dataset[0])
# print(dataset[1])
# print(dataset[2])

batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
