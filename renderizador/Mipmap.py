import numpy as np
from matplotlib import pyplot as plt
from typing import List

# Enable printing and plotting debug info
DEBUG_INFO = False

class Mipmap():
    def __init__(self, image: np.array, maxLevel: int = None) -> None:
        """
        Construct a mipmap of provided image array.
        maxLevel indicates maximum depth of bilinear reduction. 
            - 0 means no reduction is applied, 
            - None means reduce until image is 2x2
            - Value higher than int(np.log2(max(image.shape))) is ignored
        To use Mipmap, check method get_texture(x, y, L) 
        """
        # Input verification
        w, h = image.shape[:2]
        if DEBUG_INFO: print("Input image shape: ({}x{})".format(w,h))
        if int(np.log2(w)) - np.log2(w) != 0: print("WARNING! PROVIDED TEXTURE WIDTH IS NOT IN A POWER OF 2 SHAPED ARRAY!") 
        if int(np.log2(h)) - np.log2(h) != 0: print("WARNING! PROVIDED TEXTURE WIDTH IS NOT IN A POWER OF 2 SHAPED ARRAY!") 
        
        # Calculate amount of necessary levels
        self.L = int(np.log2(max(w, h)))
        if maxLevel is None: maxLevel = self.L
        
        # Dictionary to store mipmapped texture.
        self._map = {0 : image}
        
        # Construct each level
        for step in range(0, min(self.L, maxLevel)):
            # Debug prints
            if DEBUG_INFO: print("Mipmap level: {}".format(step))
        
            # Construct every halved image:
            self._map[step+1] = self.halve_image(self._map[step])            
        
        # Show plots
        if DEBUG_INFO: self.plot()

    def halve_image(self, image: np.array) -> np.array:
        """
        Halves image size using bilinear filtering.
        Found at: https://stackoverflow.com/questions/14549696/mipmap-of-image-in-numpy
        """
        rows, cols, planes = image.shape
        image = image.astype('uint16')
        image = image.reshape(rows // 2, 2, cols // 2, 2, planes)
        image = image.sum(axis=3).sum(axis=1)
        return ((image + 2) >> 2).astype('uint8')
    
    def plot(self):
        fig = plt.figure(figsize=(9,(self.L//3)*3))
        axList = []
        for step in self._map:
            axList.append(fig.add_subplot(self.L//3+1, 3, step+1))
            axList[step].imshow(self._map[step], interpolation='nearest')
            axList[step].get_xaxis().set_visible(False)
            axList[step].get_yaxis().set_visible(False)
            axList[step].set_title("Mipmap level: {} ({}x{})".format(step, *self._map[step].shape[:2]))
        plt.show()

    def get_texture(self, u: int, v: int, L: int = 0) -> List[np.uint8]:
        """
        Returns RGB value at position x,y for a given Level of mipmap.
        """
        # L out of bounds
        if L>max(self._map.keys()): L = max(self._map.keys())
        if L<0                    : L = 0
        
        # Re-indexing to new size
        new_shape = self._map[L].shape
        j = int(u*new_shape[0])
        i = new_shape[1] - int(v*new_shape[1]) - 1
        
        # Debug printing
        if DEBUG_INFO: 
            print("Getting Mipmap texture at position ({},{}), for level {}.".format(i, j, L))
            t = self._map[L][i, j].copy()
            self._map[L][i, j] = [255,255,255]
            return t
        
        return self._map[L][i, j]

if __name__ == "__main__":
    # Black image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    # Artifacts to check functionality
    image[30:128 , 30:128 ] = [255, 0  , 0  ]
    image[200:225, 20:100 ] = [0  , 255, 0  ]
    image[20:100 , 200:225] = [0  , 0  , 255]
    image[200:225, 200:225] = [0  , 255, 255]
    m = Mipmap(image)
    print("Testing gets:")
    for i in range(9):
        print(m.get_texture(0.2, 0.2, i))
    m.plot()
