#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import chainer
import chainer.functions as cf
import chainer.links as cl
import cupy as cp
import numpy as np
import scipy.misc

sys.path.append('../')
import lib


class Scene(lib.Scene):
    def __init__(self, num_cameras, filename, texture_size, lr_vertices, light_on, init_bias):
        super(Scene, self).__init__(num_cameras=num_cameras, directional_light=light_on, ambient_light=light_on)
        if init_bias.startswith('['):
            init_bias = np.array([float(v) for v in init_bias[1:-1].split(',')], 'float32')
        else:
            init_bias = np.array(float(init_bias), 'float32')
        with self.init_scope():
            mesh = lib.ObjFile(filename=filename, texture_size=texture_size, lr_vertices=lr_vertices)
            if init_bias.ndim == 0:
                mesh.textures.data += init_bias
            else:
                mesh.textures.data += init_bias[None, None, None, None, :]
            self.add_mesh('object', mesh)


def extract_feature(googlenet, images):
    mean = cp.array([104.0, 116.0, 122.0], 'float32')  # BGR
    images = images[:, ::-1] * 255 - mean[None, :, None, None]
    features = googlenet(images, layers=['inception_4c']).values()[0]
    return features


def get_loss(scene, googlenet, images, masks, lambda_length, without_mask):
    features = extract_feature(googlenet, images)
    if without_mask:
        loss = -cf.sum(cf.square(features))
        loss /= features.size
    else:
        scale = masks.shape[2] / features.shape[2]
        masks = cf.average_pooling_2d(masks[:, None, :, :], scale, scale)[:, 0, :, :]
        loss = -cf.sum(cf.square(features * cf.broadcast_to(masks[:, None, :, :], features.shape)))
        loss /= features.shape[0] * features.shape[1] * cf.sum(masks)
    for mesh in scene.mesh_list.values():
        loss += lambda_length * mesh.get_var_line_length_loss()
    return loss


def save_image(directory, filename, scene, image_size, distance, azimuth=0):
    # set camera
    batch_size = scene.num_cameras
    azimuth_batch = cp.ones(batch_size, 'float32') * azimuth
    distance_batch = cp.ones(batch_size, 'float32') * distance
    scene.camera.set_eye(azimuth=azimuth_batch, distance=distance_batch)

    # rasterization & save
    images = scene.rasterize(image_size=image_size * 2, background_colors=1., fill_back=True).data.get()
    images = cf.resize_images(images, (image_size, image_size))
    image = images[0].transpose((1, 2, 0)).data
    image = (image * 255).clip(0., 255.).astype('uint8')
    scipy.misc.imsave(os.path.join(directory, filename), image)


def run():
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-of', '--obj_filename', type=str)
    parser.add_argument('-od', '--output_directory', type=str)
    parser.add_argument('-l', '--light', type=bool, default=False)
    parser.add_argument('-bc', '--background_color', type=float, default=1.)
    parser.add_argument('-ib', '--init_bias', type=str, default='0')
    parser.add_argument('-ll', '--lambda_length', type=float, default=1e-4)
    parser.add_argument('-emax', '--elevation_max', type=float, default=30.)
    parser.add_argument('-emin', '--elevation_min', type=float, default=-10.)
    parser.add_argument('-amax', '--azimuth_max', type=float, default=180.)
    parser.add_argument('-amin', '--azimuth_min', type=float, default=-180.)
    parser.add_argument('-al', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-lrv', '--lr_vertices', type=float, default=0.005)
    parser.add_argument('-is', '--image_size', type=int, default=224)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.5)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-ts', '--texture_size', type=int, default=4)
    parser.add_argument('-dn', '--distance_noise', type=float, default=0.1)
    parser.add_argument('-ni', '--num_iteration', type=int, default=1000)
    parser.add_argument('-ns', '--num_save', type=int, default=100)
    parser.add_argument('-wm', '--without_mask', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # init output directory
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # setup chainer
    chainer.cuda.get_device_from_id(args.gpu).use()
    cp.random.seed(0)
    np.random.seed(0)

    # load feature extractor
    googlenet = cl.GoogLeNet()
    googlenet.to_gpu()

    # setup scene & optimizer
    scene = Scene(args.batch_size, args.obj_filename, args.texture_size, args.lr_vertices, args.light, args.init_bias)
    scene.to_gpu()
    optimizer = lib.Adam(alpha=args.adam_lr, beta1=args.adam_beta1)
    optimizer.setup(scene)

    # save initial model
    filename = 'train_%08d.png' % 0
    save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance)

    # main loop
    num_iter_save = args.num_iteration / args.num_save
    for i in range(args.num_iteration):
        # setup camera
        azimuth_batch = cp.random.uniform(args.azimuth_min, args.azimuth_max, size=args.batch_size)
        azimuth_batch[azimuth_batch.size / 2:] += 180
        elevation_batch = cp.random.uniform(args.elevation_min, args.elevation_max, size=args.batch_size)
        distance_batch = cp.random.uniform(-args.distance_noise, args.distance_noise).astype('float32')
        distance_batch += args.camera_distance
        scene.camera.set_eye(azimuth_batch, elevation_batch, distance_batch)

        # get loss
        background_color = cp.random.uniform(0., 1.).astype('float32')
        images = scene.rasterize(image_size=args.image_size, background_colors=background_color, fill_back=True,
                                 anti_aliasing=True)
        masks = scene.rasterize_silhouette(image_size=args.image_size, fill_back=True, anti_aliasing=True)
        loss = get_loss(scene, googlenet, images, masks, args.lambda_length, args.without_mask)

        # update
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        # show progress
        if (i + 1) % num_iter_save == 0:
            num = (i + 1) / num_iter_save
            filename = 'train_%08d.png' % num
            save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance)
        print 'iter: %08d / %08d,' % (i, args.num_iteration), 'loss: %.4f' % float(loss.data)

    # save turntable images
    for azimuth in range(0, 360, 2):
        filename = 'rotation_%08d.png' % azimuth
        save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance, azimuth)

        # directory = args.output_directory
        # options = ' -layers optimize -loop 0 -delay 4'
        # subprocess.call('convert %s %s/rotation_*.png %s/rotation.gif' % (options, directory, directory), shell=True)
        # subprocess.call('convert %s %s/train_*.png %s/train.gif' % (options, directory, directory), shell=True)
        # convert -layers optimize -loop 0 -delay 4 rotation_*.png rotation.gif


if __name__ == '__main__':
    run()
