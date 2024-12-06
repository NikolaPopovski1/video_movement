import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.registration as skreg

# nalaganje in predvajanje z matplotlib
pot_do_posnetka = pathlib.Path('./primeri/synth_vid2/')
slike_poti = pot_do_posnetka.glob('frame*.jpg')
# te posnetke lahko uredimo po abecednem vrstnem redu, da dobimo ustrezno video sekvenco
slike_poti = sorted(slike_poti)
plt.figure()
video_seq = []
for pot in slike_poti:
    slika = plt.imread(pot)
    video_seq.append(slika)

    plt.clf()
    plt.imshow(slika)
    plt.draw()
    plt.waitforbuttonpress(0.01)
    
plt.close('all')
video = np.array(video_seq)