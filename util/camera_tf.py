import tensorflow as tf
import numpy as np

def get_ray_map(images: tf.Tensor, intrinsics: tf.Tensor, extrinsics: tf.Tensor):
    B, H, W, C = tf.unstack(tf.shape(images))

    x, y = tf.meshgrid(tf.range(W), tf.range(H))
    g = tf.stack([x, -y], axis=-1)[tf.newaxis]
    g = tf.cast(g, tf.float32)
    g = tf.tile(g, [B, 1, 1, 1])

    image_coords = g - tf.reshape([tf.cast(W, g.dtype)/2.,-tf.cast(H, g.dtype)/2.], [1,1,1,2])

    # homo coords B x H x W x 3
    image_hcoords = tf.concat([image_coords, -tf.ones([B, H, W, 1], dtype=image_coords.dtype)], axis=3)

    # transposed homo coords B x 3 x (HxW)
    image_hcoords_t = tf.transpose(tf.reshape(image_hcoords, [B, H*W, 3]), [0,2,1])

    # extrinsics: B x 4 x 4
    extrinsics = tf.reshape(extrinsics, [B, 4, 4])


    rays_d = tf.matmul(tf.linalg.inv(intrinsics[:, :3, :3]), image_hcoords_t)
    rays_d = tf.matmul(extrinsics[:,:3, :3], rays_d)  # (B x 3 x H*W)
    rays_d = tf.transpose(rays_d, (0, 2, 1))  # (B, H*W, 3)
    rays_d = tf.reshape(rays_d, (B, H, W, 3))

    rays_o = tf.reshape(extrinsics[:, :3, 3], (B, 1, 3))
    rays_o = tf.tile(rays_o, (1, H*W, 1))  # (B, H*W, 3)
    rays_o = tf.reshape(rays_o, [B, H, W, 3])

    return rays_o, rays_d


def get_intersaction_sphere(rays_o, rays_d, center, radius, only_forward=True, get_mask=False):
    n_rays_o = rays_o - center
    n_rays_d = rays_d / tf.linalg.norm(rays_d, axis=-1, keepdims=True)
    a = tf.ones_like(rays_d)[...,:1]
    b = 2.0 * tf.reduce_sum(n_rays_o * n_rays_d, axis=-1, keepdims=True)
    c = tf.reduce_sum(tf.square(n_rays_o), axis=-1, keepdims=True) - radius * radius


    k = b*b - 4.*a*c
    valid = tf.cast(k >= 0., k.dtype)
    pm = tf.math.sqrt(k * valid)

    oc = center - rays_o

    _nm = lambda x: tf.linalg.norm(x, axis=-1, keepdims=True)
    cos_t = tf.reduce_sum(n_rays_d * oc, axis=-1, keepdims=True)
    cos_t = tf.math.divide_no_nan(cos_t, _nm(oc))
    theta = tf.math.acos(tf.clip_by_value(cos_t, 0., 1.))
    phi = tf.acos(_nm(oc) * tf.math.sin(theta) / radius)
    i1 = rays_o + (_nm(oc) * tf.cos(theta) + tf.sin(phi) * radius) * n_rays_d


    if only_forward:
        if get_mask:
            return i1, valid
        else:
            return i1
    else:
        i2 = rays_o + (_nm(oc) * tf.cos(theta) - tf.sin(phi) * radius) * n_rays_d
        if get_mask:
            return i1, i2, valid
        else:
            return i1, i2

def get_reparam_point(rays_o, pnts_a, rays_d, one_over_r, center, radius):
    _nm = lambda x: tf.linalg.norm(x, axis=-1, keepdims=True)

    one_over_r = one_over_r / radius

    oa = rays_d
    ca = pnts_a - center

    cos_t = tf.reduce_sum(oa * ca, axis=-1, keepdims=True) / _nm(oa) / _nm(ca)
    theta = tf.math.acos(tf.clip_by_value(cos_t, 0., 1.))
    phi = np.pi/2. - theta

    gamma = tf.acos(radius * tf.cos(phi)[...,None,:] * one_over_r[...,:,None]) - phi[...,None,:]

    k = tf.linalg.cross(ca,oa)
    nmk = _nm(k)
    k = tf.math.divide_no_nan(k,nmk)
    v_t = pnts_a[...,None,:] * tf.cos(gamma) \
          + tf.linalg.cross(k,pnts_a)[...,None,:]*tf.sin(gamma) \
          + (k * tf.reduce_sum(k*pnts_a, axis=-1, keepdims=True))[...,None,:] * (1.-tf.cos(gamma))
    ref_vt = tf.zeros_like(v_t)

    viewdir = rays_d / tf.linalg.norm(rays_d, axis=-1, keepdims=True)
    cond = ((_nm(k)[...,None,:] + ref_vt) == 0.)
    v_t = tf.where(cond, (center + viewdir*radius)[...,None,:] + ref_vt, v_t)
    #print(tf.reduce_sum(tf.cast(cond, tf.int32)), rays_d.shape)

    return v_t
