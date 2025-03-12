import pandas as pd
import numpy as np

m1 = np.random.rand(5, 4)
m2 = np.random.rand(5, 3)

df1 = pd.DataFrame(m1, columns=['Col1', 'Col2', 'Col3', 'Col4'])

df2 = pd.DataFrame(m2, columns=['Col5', 'Col6', 'Col7'])

merged = pd.concat([df1, df2], axis=1)

print(merged)
