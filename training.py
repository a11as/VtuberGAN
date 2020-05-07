import tqdm
import torch
from statistics import mean
from torch import nn, optim
from torchvision.utils import save_image

from import_all_dataset import batch_size, data_loader
from models import Generator, Discriminator

# 利用するエンジンを決定
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_G = Generator().to(device)
model_D = Discriminator().to(device)

params_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
params_D = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 潜在特徴100次元ベクトルz
nz = 100


# ロスを計算するときのラベル変数
ones = torch.ones(batch_size).to(device) # 正例 1
zeros = torch.zeros(batch_size).to(device) # 負例 0
loss_f = nn.BCEWithLogitsLoss()

# 潜在特徴z
check_z = torch.randn(batch_size, nz, 1, 1).to(device)

# 訓練関数
def train_dcgan(model_G, model_D, params_G, params_D, data_loader):
    log_loss_G = []
    log_loss_D = []
    for real_img, _ in tqdm.tqdm(data_loader):
        batch_len = len(real_img)

        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, nz, 1, 1).to(device)
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 注(１)
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()


        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.to(device)

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算省略 => 注（１）
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

    return mean(log_loss_G), mean(log_loss_D)

for epoch in range(300):
    train_dcgan(model_G, model_D, params_G, params_D, data_loader)

    if epoch % 10 == 0:
        torch.save(model_G.state_dict(), "./output/Weight_G{:03d}.prm".format(epoch), pickle_protocol=4)

        torch.save(model_D.state_dict(), "./output/Weight_D{:03d}.prm".format(epoch), pickle_protocol=4)

        generated_img = model_G(check_z) 
        save_image(generated_img, "./output/Generated_Image/{:03d}.jpg".format(epoch))