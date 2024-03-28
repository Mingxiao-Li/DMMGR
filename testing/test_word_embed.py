import numpy as np


if __name__ == "__main__":
    path = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/glove.42B.300d/glove.42B.300d.npy"
    word_embed = np.load(path)
    print(word_embed.size)