import numpy as np
import matplotlib.pyplot as plt

points = [(10,10), (20,20), (30,30)]
thetaScale = np.linspace(0, np.pi, 500)
rhoScale = np.zeros((3,500))

for i in range(len(points)):
    for j in range(len(thetaScale)):
        rhoScale[i,j] = points[i][1] * np.cos(thetaScale[j]) + points[i][0] * np.sin(thetaScale[j])

plt.plot(thetaScale, rhoScale[0,:], label='(10,10)')
plt.plot(thetaScale, rhoScale[1,:], label='(20,20)')
plt.plot(thetaScale, rhoScale[2,:], label='(30,30)')
plt.plot(np.linspace(3*np.pi/4, 3*np.pi/4, 10), np.linspace(-40, 0, 10), linestyle='dotted', color='grey')
plt.plot(np.linspace(0, 3*np.pi/4, 10), np.linspace(0, 0, 10), linestyle='dotted', color='grey')
plt.xlabel('\u03B8')
plt.ylabel('\u03C1')
plt.legend()
plt.show()
