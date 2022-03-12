import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pathlib import Path
from render_functors import *

fig = plt.figure()
ax = fig.add_subplot(111)

input_meta = load_input()
nfile = get_framenumber(input_meta)
dfunc, afunc = draw_machnumber(input_meta)

# I like to position my colorbars this way, but you don't have to
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

# This is now a list of arrays rather than a list of artists
frames = []
for i in range(nfile):
    frames.append(dfunc(i))
    print("Render frame %d" % (i,))

cv0 = frames[0]
im = ax.imshow(cv0, origin='lower')  # Here make an AxesImage rather than contour
cb = fig.colorbar(im, cax=cax)
#tx = ax.set_title('Frame 0')
cb.set_label(afunc())
ax.set_xticks([])
ax.set_yticks([])

def animate(i):
    arr = frames[i]
    vmax = np.max(arr)
    vmin = np.min(arr)
    im.set_data(arr)
    im.set_clim(vmin, vmax)
    #tx.set_text('Frame {0}'.format(i))
    # In this version you don't have to do anything to the colorbar,
    # it updates itself when the mappable it watches (im) changes



ani = animation.FuncAnimation(fig, animate, frames=nfile, repeat_delay=1000)

plt.show()
