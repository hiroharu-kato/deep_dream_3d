#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as cf
import chainer.links as cl
import neural_renderer


def get_var_line_length_loss(vertices, faces):
    vertices = vertices[faces]
    num_faces = vertices.shape[0]
    v01 = vertices[:, 1] - vertices[:, 0]
    v12 = vertices[:, 2] - vertices[:, 1]
    v20 = vertices[:, 0] - vertices[:, 2]
    n01_square = cf.sum(cf.square(v01), axis=1)
    n12_square = cf.sum(cf.square(v12), axis=1)
    n20_square = cf.sum(cf.square(v20), axis=1)
    n01 = cf.sqrt(n01_square)
    n12 = cf.sqrt(n12_square)
    n20 = cf.sqrt(n20_square)
    mean_of_square = (cf.sum(n01_square) + cf.sum(n12_square) + cf.sum(n20_square)) / (3. * num_faces)
    square_of_mean = cf.square((cf.sum(n01) + cf.sum(n12) + cf.sum(n20)) / (3. * num_faces))
    return (mean_of_square - square_of_mean) * num_faces


class DeepDreamModel(chainer.Chain):
    def __init__(
            self,
            filename_obj,
            texture_size=4,
            init_bias=(0, 0, 0),
            image_size=224,
            layer_name='inception_4c',
            camera_distance=2.732,
            camera_distance_noise=0.1,
            elevation_min=-10,
            elevation_max=30,
            lr_vertices=0.01,
            lr_textures=1.0,
            lambda_length=0.0001,
    ):
        super(DeepDreamModel, self).__init__()
        self.image_size = image_size
        self.layer_name = layer_name
        self.camera_distance = camera_distance
        self.camera_distance_noise = camera_distance_noise
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.lambda_length = lambda_length

        self.googlenet = cl.GoogLeNet()

        with self.init_scope():
            # load .obj
            self.mesh = neural_renderer.Mesh(filename_obj, texture_size)
            self.mesh.set_lr(lr_vertices, lr_textures)

            # texture bias
            init_bias = self.xp.array(init_bias, 'float32')
            self.mesh.textures.data += init_bias[None, None, None, None, :]

            # setup renderer
            renderer = neural_renderer.Renderer()
            renderer.light_intensity_directional = 0.0
            renderer.light_intensity_ambient = 1.0
            renderer.image_size = image_size
            self.renderer = renderer

    def to_gpu(self):
        super(DeepDreamModel, self).to_gpu()
        self.googlenet.to_gpu()

    def extract_feature(self, images):
        mean = self.xp.array([104.0, 116.0, 122.0], 'float32')  # BGR
        images = images[:, ::-1] * 255 - mean[None, :, None, None]
        features = self.googlenet(images, layers=[self.layer_name]).values()[0]
        return features

    def __call__(self, batch_size):
        xp = self.xp

        # set random background color
        # background_color = xp.random.uniform(0., 1., size=(batch_size, 3)).astype('float32')
        # background_color = xp.random.uniform(0., 1., size=(batch_size,)).astype('float32')
        # background_color = xp.tile(background_color[:, None], (1, 3))
        background_color = xp.ones((batch_size, 3), 'float32') * xp.random.uniform(0., 1.)
        self.renderer.background_color = background_color

        # set random viewpoints
        self.renderer.eye = neural_renderer.get_points_from_angles(
            distance=(
                xp.ones(batch_size, 'float32') * self.camera_distance +
                xp.random.normal(size=batch_size).astype('float32') * self.camera_distance_noise),
            elevation=xp.random.uniform(self.elevation_min, self.elevation_max, batch_size).astype('float32'),
            azimuth=xp.random.uniform(0, 360, size=batch_size).astype('float32'))

        # compute loss
        images = self.renderer.render(*self.mesh.get_batch(batch_size))
        features = self.extract_feature(images)
        loss = -cf.sum(cf.square(features)) / features.size

        var_line_length = get_var_line_length_loss(self.mesh.vertices, self.mesh.faces)
        loss += self.lambda_length * var_line_length

        return loss
