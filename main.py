from convert_data_brain import convert_data_brain
from network import UnetWithHeader
from simclr import train_simclr
from unet1 import load_path, train_unet


def main():
    print("Converting data...")
    convert_data_brain()
    print("Done converting.")
    print("Training simclr...")
    train_simclr()
    print("Done training simclr.")
    print("Training unet...")
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="mlp")
    model = model.cuda()
    model = load_path(model, "./results/unet_simclr.pth")
    train_unet(model)
    print("Done training unet.")


if __name__ == '__main__':
    main()
