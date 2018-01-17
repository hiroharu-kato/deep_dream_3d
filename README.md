# 3D DeepDream using Neural Renderer

This is the code for 3D DeepDream in the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.

Related repositories:
* [Neural Renderer](https://github.com/hiroharu-kato/neural_renderer)
    * single-image 3D mesh reconstruction
    * 2D-to-3D style transfer
    * 3D DeepDream (this repository)

## Installation
```
# install neural_renderer
git clone https://github.com/hiroharu-kato/neural_renderer.git
cd neural_renderer
python setup.py install --user
# or, sudo python setup.py install
```

## Run example
```

bash ./examples/run.sh
```

![](https://raw.githubusercontent.com/hiroharu-kato/deep_dream_3d/master/examples/data/teapot.gif)
![](https://raw.githubusercontent.com/hiroharu-kato/deep_dream_3d/master/examples/data/bunny.gif)



## Citation

```
@article{kato2017renderer,
  title={Neural 3D Mesh Renderer},
  author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv:1711.07566},
  year={2017}
}
```