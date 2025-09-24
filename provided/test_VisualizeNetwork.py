import numpy as np
import matplotlib.pyplot as plt
from VisualizeNetwork import VisualizeNetwork

print('\n******* test_VisualizeNetwork.py ***********')
print('\nExample showing how to use VisualizeNetwork.py')

NodeCoordinates = np.array([[0, 0], [3, 4], [0, 7], [6, 4]])
NodalValues = np.array([1, 0.5, 1.5, 0.3])  # use here instead the solution from your simulator

EdgeConnections = np.array([[1, 2], [2, 3], [3, 1], [2, 4]])  # you can derive this from your nodal matrix
FlowValues = np.array([6, 3, 10, 1])  # you can compute this using the consitutite equations

print("NodeCoordinates =", NodeCoordinates)
print("NodalValues =", NodalValues)
print("EdgeConnections =", EdgeConnections)
print("FlowValues =", FlowValues)

# Visualize the network
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

VisualizeNetwork(ax, NodeCoordinates, NodalValues, EdgeConnections, FlowValues)
# plt.axis('equal') # this makes circles look like circles rather than ovals
# plt.draw()
plt.show()

# if you need to visualize multiple networks at the same time 
# (e.g. electrical, thermal, fluid, etc...)
# you can use a different color and height for each network
# if you do not specify the color the dafault will be 'b' for blue
# if you do not specify the height the default will be 0
# here is an example of how to visualize a second network

NodeCoordinatesNetB = np.array([[0, 7], [5, 8], [-3, 13]])
NodalValuesNetB = np.array([0.5, 1, 0.5])

EdgeConnectionsNetB = np.array([[1, 2], [1, 3]])
FlowValuesNetB = np.array([4, 3])

print("FlowValuesNetB =", FlowValuesNetB)

color = 'r'
height = 1


fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

VisualizeNetwork(ax2, NodeCoordinatesNetB, NodalValuesNetB, EdgeConnectionsNetB, FlowValuesNetB, color, height)
plt.show()

input('press a key to continue...')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')  # Add 3D projection
ax3.view_init(elev=25, azim=30)

print('and even make them rotate')
for theta in np.arange(30, 210, 1):
    ax3.view_init(elev=25, azim=theta)
    plt.draw()
    plt.pause(0.01)
  
input('press a key to continue...')
print('Example showing how to use make animations with VisualizeNetwork.py')

NodeCoordinates = np.array([[0, 0], [3, 4], [0, 7]])
EdgeConnections = np.array([[1, 2], [2, 3], [3, 1]]) # you can derive this from your nodal matrix

# here I am generating some fake signals just to illustrate how to visualize
# the behaviour of your network using VisualizeNetwork in an animation

plt.ion()  # Turn on interactive mode for animation
fig = plt.figure()
ax4 = fig.add_subplot(111, projection='3d')

N = 50
for t in np.linspace(0, 2 * np.pi * 2, N):
    ax4.cla()  # Clear the axis for new plot
    NodalValues = np.array([1, 0.5, 1.5]) * (0.4 * np.cos(t) + 1) * 0.8
    FlowValues = np.array([8, 5, 15]) * (0.4 * np.sin(t) + 1) * 0.2
    VisualizeNetwork(ax4, NodeCoordinates, NodalValues, EdgeConnections, FlowValues)
    ax4.set_xlim([-4, 5])
    ax4.set_ylim([-3, 11])
    ax4.set_zlim([0, 2])

    plt.draw()
    plt.pause(0.05)  # Animation speed

plt.ioff()  # Turn off interactive mode
plt.show()