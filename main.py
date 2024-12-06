import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.registration as skreg

#"""
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    cv2.polylines(img_bgr, lines, isClosed=False, color=(0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    return img_bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ((ang * (180 / np.pi / 2)) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 16, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_bgr
#"""

video_capture = cv2.VideoCapture('./primeri/synth_vid4/%4d.jpg')
#video_capture = cv2.VideoCapture('./primeri/synth_vid4/%4d.jpg')
""" # Manual load video
image_folder = pathlib.Path('./primeri/vid1/')
image_paths = sorted(image_folder.glob('frame_*.jpg'))
video_seq = []
for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        continue
    video_seq.append(img)
if not video_seq:
    raise ValueError("No images were loaded. Check the file path and naming pattern.")
"""
#""" # Load video with cv2.VideoCapture
video_seq = []
while True:
    f, img = video_capture.read()
    if not f:
        break
    video_seq.append(img)
    # prikaz
    # cv2.imshow('video', img)
    # cv2.waitKey(10)
# zapremo okno
cv2.destroyAllWindows()
#"""

video = np.array(video_seq)[:, :, :, (2, 1, 0)] # BGR v RGB
#video = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video_seq]) #grayscale

video = video.astype(np.float32) / 255
slika_bg = video.mean(0)

#""" # OpenCV bg removal
bg_sub_obj = cv2.createBackgroundSubtractorMOG2()

plt.figure()
slikaPrev = np.uint8(video[0]*255)
for n in range(video.shape[0]):
    slika = np.uint8(video[n]*255)
    # opencv je malo razvajen, hoče imeti zelo specifične tipe, da deluje pravilno
    slika_motion_seg_prev = bg_sub_obj.apply(slikaPrev)
    slika_motion_seg = bg_sub_obj.apply(slika)
    slika_motion_bg = bg_sub_obj.getBackgroundImage()
    flow = cv2.calcOpticalFlowFarneback(slika_motion_seg_prev, slika_motion_seg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    slikaPrev = slika
    plt.clf()
    plt.subplot(1,2,1)
    #plt.imshow(slika)
    plt.imshow(draw_flow(slika, flow))
    plt.title('posnetek')
    # plt.subplot(1,3,2)
    # plt.imshow(slika_motion_bg)
    # plt.title('slika ozadja')
    plt.subplot(1,2,2)
    #plt.imshow(slika_motion_seg)
    plt.imshow(draw_hsv(flow))
    plt.title('segmenti gibanja')
    #plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')
#"""

""" # Bg removal
n=31
slika = video[n]
slika_diff = np.abs(slika - slika_bg).mean(2) # povprecje po barvah, ce imamo barve
#slika_diff = np.abs(slika - slika_bg) # povprecje po grayscale

plt.figure()
plt.subplot(221)
plt.imshow(slika_bg)
plt.title('ozadje')
plt.axis('off')
plt.subplot(222)
plt.imshow(slika)
plt.title(f'slika {n}')
plt.axis('off')
plt.subplot(223)
plt.imshow(slika_diff)
#plt.imshow(slika_diff, cmap='gray') # če želimo sivinsko sliko
plt.title('slika razlik')
plt.axis('off')
plt.subplot(224)
plt.hist(slika_diff.ravel(), bins=100, log=True)
plt.title('histogram slike razlik')
prag = 0.4

plt.figure()
for n in range(video.shape[0]):
    slika = video[n]
    slika_diff = np.abs(slika - slika_bg).mean(2) # povprecje po barvah, ce imamo barve
    #slika_diff = np.abs(slika-slika_bg) # povprecje po grayscale
    slika_motion_seg = slika_diff>prag

    plt.clf()
    plt.imshow(slika_motion_seg)
    #plt.imshow(slika_motion_seg, cmap='gray') # če želimo sivinsko sliko
    plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')
"""