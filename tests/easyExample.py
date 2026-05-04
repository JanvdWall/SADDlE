from SADDlE import generalDDE, assembleAm
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import tikzplotlib

dde = generalDDE(1, [], [1], [0], np.array([[-150]]), [], [lambda theta: np.array([[-20]])]).convertToTheoreticalDDE()
Am = assembleAm(dde, 300, useSparse=True)
eigenValues = np.sort_complex(eig(Am.toarray())[0])[-30::]
AmSmall = assembleAm(dde, 20, useSparse=True)
eigenValuesSmall = np.sort_complex(eig(AmSmall.toarray())[0])[-14::]
AmMed = assembleAm(dde, 50, useSparse=True)
eigenValuesMed = np.sort_complex(eig(AmMed.toarray())[0])[-20::]


x = [val.real for val in eigenValues]
y = [val.imag for val in eigenValues]

xSmall = [val.real for val in eigenValuesSmall]
ySmall = [val.imag for val in eigenValuesSmall]

xMed = [val.real for val in eigenValuesMed]
yMed = [val.imag for val in eigenValuesMed]


# plot the complex numbers
plt.scatter(x, y, marker='o', label='$M=300')
plt.scatter(xMed, yMed, marker='v', label='$M=80')
plt.scatter(xSmall, ySmall, marker='x', label='$M=20')

plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.legend()
tikzplotlib.save("easyExample.tex")