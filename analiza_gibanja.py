import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.registration as skreg

"""
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
"""

#video_capture = cv2.VideoCapture('./primeri/synth_vid4/%4d.jpg')
#video_capture = cv2.VideoCapture('./primeri/synth_vid2/frame.%4d.jpg')
#""" # Manual load video
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
#"""
""" # Load video with cv2.VideoCapture
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
"""

video = np.array(video_seq)[:, :, :, (2, 1, 0)]
#video = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video_seq]) #grayscale

video = video.astype(np.float32) / 255



slika_bg = video.mean(0)

prag = 0.22 # synt_vid1 -> 0.1, synt_vid2 -> 0.29, synt_vid3 -> 0.042, synt_vid4 -> 0.08, vid1 -> 0.22, vid2 -> 0.205

video_diff = []
#plt.figure()
for n in range(video.shape[0]):
    slika = video[n]
    slika_diff = np.abs(slika-slika_bg).mean(2) 
    slika_motion_seg = slika_diff>prag
    video_diff.append(slika_motion_seg)
"""
    #plt.clf()
    #plt.subplot(1,2,1)
    #plt.imshow(slika_motion_seg)
    #plt.title('slika ozadja')
    #plt.subplot(1,2,2)
    #plt.imshow(slika)
    #plt.title('slika ozadja')
    #plt.draw()
    #plt.waitforbuttonpress(0.01)

#plt.close('all')
"""


""" #almost working bg removal
bg_sub_obj = cv2.createBackgroundSubtractorMOG2()

# Initialize background image as a floating-point array
buki = None

# Process each frame and update the background
plt.figure()
for n in range(video.shape[0]):
    slika = np.uint8(video[n] * 255)  # Convert video frame to 8-bit
    slika_motion_seg = bg_sub_obj.apply(slika)  # Segment moving parts
    slika_motion_bg = bg_sub_obj.getBackgroundImage()  # Background image
    
    # Convert background image to the same scale as `video`
    if slika_motion_bg is not None:
        buki = slika_motion_bg.astype(np.float32) / 255  # Normalize to 0-1 range

# Ensure buki is available
if buki is None:
    raise ValueError("Background model failed to initialize properly.")

# Threshold and display differences
prag = 0.1
for n in range(video.shape[0]):
    slika = video[n]
    slika_diff = np.abs(slika - buki).mean(2)  # Average over color channels
    slika_motion_seg = slika_diff > prag

    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(slika_motion_seg)
    plt.title('slika izreza')
    plt.subplot(1,3,2)
    plt.imshow(slika)
    plt.title('slika ozadja')
    plt.subplot(1,3,3)
    plt.imshow(buki)
    plt.title('slika ozadja')
    plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')
"""

#"""
video_diff = np.array(video_diff)

# Optical flow visualization
plt.figure()
for n in range(video_diff.shape[0] - 1):
    slika_0 = video_diff[n]
    slika_1 = video_diff[n + 1]

    # Convert frames to grayscale
    #slika_0_gray = cv2.cvtColor((slika_0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #slika_1_gray = cv2.cvtColor((slika_1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    slika_0_gray = (slika_0 * 255).astype(np.uint8)
    slika_1_gray = (slika_1 * 255).astype(np.uint8) 

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(slika_0_gray, slika_1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    opt_flow_vis_hsv = np.zeros(mag.shape + (3,), dtype=np.uint8)
    opt_flow_vis_hsv[:, :, 0] = ang * 180 / np.pi / 2
    opt_flow_vis_hsv[:, :, 1] = 255
    opt_flow_vis_hsv[:, :, 2] = (mag/mag.max()) ** 0.5 * 255
    opt_flow_vis = cv2.cvtColor(opt_flow_vis_hsv, cv2.COLOR_HSV2RGB)
    
    plt.clf()
    plt.subplot(221)
    plt.imshow(video[n])
    plt.subplot(222)
    plt.imshow(slika_1, cmap='gray')
    plt.subplot(223)
    plt.imshow(opt_flow_vis)

    plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')



"""
plt.clf()
plt.subplot(1,1,1)
plt.imshow(buki)
plt.title('posnetek')

plt.draw()
plt.show()
"""
"""
plt.close('all')
noBgVideo = np.array(noBgVideoSeq)
for n in range(noBgVideo.shape[0]):
    slika = np.uint8(noBgVideo[n]*255)
    
    plt.clf()
    plt.subplot(1,1,1)
    plt.imshow(slika)
    plt.title('posnetek')
    
    plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')
"""
"""
slika_bg = video.mean(0)

n=31
slika = video[n]
slika_diff = np.abs(slika-slika_bg).mean(2) # povprecje po barvah, ce imamo barve

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
plt.title('slika razlik')
plt.axis('off')
plt.subplot(224)
plt.hist(slika_diff.ravel(), bins=100, log=True)
plt.title('histogram slike razlik')
prag = 0.4

plt.figure()
for n in range(video.shape[0]):
    slika = video[n]
    slika_diff = np.abs(slika-slika_bg).mean(2) 
    slika_motion_seg = slika_diff>prag

    plt.clf()
    plt.imshow(slika_motion_seg)
    plt.draw()
    plt.waitforbuttonpress(0.01)

plt.close('all')
"""