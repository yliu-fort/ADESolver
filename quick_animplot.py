import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pathlib import Path
from render_functors import *

fig, ax = plt.subplots()

input_meta = load_input()
nfile = get_framenumber(input_meta)
dfunc, _ = draw_density(input_meta)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
start = 0
for i in range(start, nfile):
    im = ax.imshow(dfunc(i), animated=True)
    print("Render frame %d" % (i,))
    if i == start:
        ax.imshow(dfunc(start))  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
