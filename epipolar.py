import os
import tensorflow as tf
import cv2
import numpy as np
import data.nerf_pp
import data.nerf_synthetic
os.environ["CUDA_VISIBLE_DEVICES"] = ""



#d = data.nerf_synthetic.NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'train', white_bkgd=True,)
d = data.nerf_pp.NerfPPData('nerfppdata', 'scan65_paper', 'train', white_bkgd=False, normalize_coordinates=True, shuffle=False)
d.compute_inside()

d = tf.data.Dataset.zip((d.dataset, d.ray_dataset))
d = d.shuffle(50)

xs = []
ys = []
for (x, y), (xx, yy) in d:
    xs.append(yy)
    ys.append(y)

i1 = ys[0]['intrinsic']
e1 = ys[0]['c2w']
i2 = ys[1]['intrinsic']
e2 = ys[1]['c2w']

im1 = xs[0].numpy()[..., [2,1,0]]
im2 = xs[1].numpy()[..., [2,1,0]]

H, W, C = im1.shape

print(H,W,C)



def get_ray(i1, e1, coord):
    coord = np.reshape([coord[0], coord[1], -1.], [3,1])
    ray_d = np.matmul(np.linalg.inv(i1[:3, :3]), coord)
    ray_d = np.matmul(e1[:3, :3], ray_d)  # (3 x 1)
    ray_d = np.transpose(ray_d, (1, 0))  # (1 x 3)
    ray_d = np.reshape(ray_d, (3))

    ray_o = np.reshape(e1[:3, 3], (3))

    return ray_o, ray_d

def world2cam(i, c2w, point):
    w2c = np.linalg.inv(c2w)
    point = np.reshape([point[0], point[1], point[2], 1.], [4,1])
    point = np.matmul(w2c, point)
    point = point[0:3] / point[3:4]
    coord = np.matmul(i, point)
    coord = coord[0:2] / -coord[2:3]
    return coord

def draw_line(x1, x2, W, H):
    x1 = np.array(x1)
    x2 = np.array(x2)
    diff = x2 - x1
    if diff[0] == 0.:
        sp = np.array([x1[0], 0])
        ep = np.array([x1[0], H-1])
        return sp, ep
    elif diff[1] == 0.:
        sp = np.array([0., x1[1]])
        ep = np.array([W-1., x1[1]])
        return sp, ep
    else:
        slope_x = diff[1] / diff[0]
        slope_y = diff[0] / diff[1]

        x0y = slope_x * (-x1[0]) + x1[1]
        xwy = slope_x * (W-1.-x1[0]) + x1[1]

        y0x = slope_y * (-x1[1]) + x1[0]
        yhx = slope_y * (H - 1. - x1[1]) + x1[0]

        points = []
        if 0. <= x0y <= H-1:
            p = np.array([0., x0y])
            points.append(p)

        if 0. <= xwy <= H-1:
            p = np.array([W - 1., xwy])
            points.append(p)

        if 0. < y0x < W-1:
            p = np.array([y0x, 0.])
            points.append(p)

        if 0. < yhx < W-1:
            p = np.array([yhx, H-1.])
            points.append(p)

        return points

disp_down_scale = 2
_r = lambda x: cv2.resize(x, (x.shape[1] // disp_down_scale,x.shape[0] // disp_down_scale))

def click(event, x, y, flags, param):
    global im1, im2
    if event == cv2.EVENT_LBUTTONDOWN:
        H, W, C = im1.shape
        # im2 = ims[1][..., [2, 1, 0]].copy()
        im1 = im1.copy()
        im2 = im2.copy()
        x = x * disp_down_scale
        y = y * disp_down_scale
        clicked = (x  - W//2, H//2 - y)

        ray_o, ray_d = get_ray(i1, e1, clicked)
        #print(ray_o, ray_d)
        near = ray_o
        far = ray_o + ray_d

        n2 = world2cam(i2, e2, near)
        f2 = world2cam(i2, e2, far)


        print(n2, f2)

        start_point = (int(n2[0] + W//2), int(- n2[1] + H//2))
        end_point = (int(f2[0] + W//2), int(- f2[1] + H//2))

        start_point, end_point = draw_line(start_point, end_point, W, H)
        start_point = start_point.astype(np.int32)
        end_point = end_point.astype(np.int32)
        print(start_point, end_point)
        color = np.random.uniform(0, 1, size=3).tolist()
        color = tuple(map(lambda x: float(x), color))
        thickness = 3
        im1 = cv2.circle(im1, (x,y), radius=5, color=color, thickness=-1)
        im2 = cv2.line(im2, start_point, end_point, color, thickness)
        cv2.imshow("im2", _r(im2))


cv2.namedWindow("im1")
cv2.setMouseCallback("im1", click)

while True:
    cv2.imshow("im1", _r(im1))
    cv2.imshow("im2", _r(im2))

    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        xs = []
        ys = []
        for (x, y), (xx, yy) in d:
            xs.append(yy)
            ys.append(y)

        i1 = ys[0]['intrinsic']
        e1 = ys[0]['c2w']
        i2 = ys[1]['intrinsic']
        e2 = ys[1]['c2w']

        print (i1, i2)

        im1 = xs[0].numpy()[..., [2, 1, 0]]
        im2 = xs[1].numpy()[..., [2, 1, 0]]

