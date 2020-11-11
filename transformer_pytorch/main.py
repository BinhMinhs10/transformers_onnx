import numpy as np
import torch



nopeak_mask = np.triu(np.ones((1, 10, 10)), k=1).astype('uint8')
nopeak_mask = torch.from_numpy(subsequent_mask) == 0
print(nopeak_mask)