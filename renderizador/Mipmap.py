import numpy as np

class Mipmap():
    def __init__(self, image) -> None:
        w, h = image.shape[:2]
        if int(np.log2(w)) - np.log2(w) != 0: print("WARNING! PROVIDED TEXTURE WIDTH IS NOT IN A POWER OF 2 SHAPED ARRAY!") 
        if int(np.log2(h)) - np.log2(h) != 0: print("WARNING! PROVIDED TEXTURE WIDTH IS NOT IN A POWER OF 2 SHAPED ARRAY!") 
        L = int(np.log2(max(w, h)))
        _map = dict()
        for i in range(L, 0, -1):
            print(i)

    def get_l(self) -> None:
        pass

if __name__ == "__main__":
    m = Mipmap(np.random.normal(128, scale=10, size=(256,256,4)))
