import numpy as np
import matplotlib.pyplot as plt


def main():
    training_loss = np.load("outputs/training-loss0.npy")
    validation_loss = np.load("outputs/validation-loss0.npy")

    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()