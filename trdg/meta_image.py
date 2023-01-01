from PIL import Image
import numpy as np

class MetaImage:
    image: Image
    font: str
    font_size: int
    font_width: int
    
    def __init__(self, image: Image) -> None:
        self.image = image
        
        
    def image(self) -> Image:
        return self.image 
    
    def nparray(self) -> np.array:
        return np.asarray(self.image)
    
    