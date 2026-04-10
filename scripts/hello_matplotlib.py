import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 300)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.set_title('Hello, matplotlib!')
ax.legend(); plt.tight_layout(); 
plt.savefig('../docs/images/hello_matplotlib.png', transparent=True)
