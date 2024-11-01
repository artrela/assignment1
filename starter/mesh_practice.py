
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_shape(shape):

    device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=256)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    num_frames = 500
    duration = num_frames / 60

    if shape == "tet":
        points =  get_tetrahedron_points(side_len=2)  # (N_v, 3) -> (1, N_v, 3)
        faces = get_tetrahedron_faces()  # (N_f, 3) -> (1, N_f, 3)
    else:
        points =  get_square_points(side_len=1)  # (N_v, 3) -> (1, N_v, 3)
        faces = get_square_faces()  # (N_f, 3) -> (1, N_f, 3)

    
    renders = []
    for deg in tqdm(range(num_frames)):

        # x_rot, y_rot, z_rot = np.random.rand(0, 5), np.random.randint(0, 5), np.random.randint(0, 5)
        if deg > num_frames / 2:
            points = rotate_points(points, x_rot=2, y_rot=1, z_rot=5)
        else:
            points = rotate_points(points, x_rot=5, y_rot=3, z_rot=1)

        vertices = torch.from_numpy(points).unsqueeze(0).float()

        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor([[0, 1, 0.35]])  # (1, N_v, 3)

        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {i}", fill=(255, 0, 0))
        images.append(np.array(image))

    
    imageio.mimsave(f"results/{shape}.gif", images, fps=(num_frames / duration))


def rotate_points(ps, x_rot, y_rot,  z_rot):

    r = R.from_euler('zyx', [z_rot, y_rot, x_rot], degrees=True)
    ps = ps @ r.as_matrix()

    return ps


def get_tetrahedron_faces():

    return torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]).unsqueeze(0)


def get_tetrahedron_points(side_len:float):
    
    p = side_len * (3**0.5) / 3

    ps = np.array([
        [0, 0, 0.5*p],
        [0, 0.5*p, -0.5*p],
        [0.5*p, -0.5*p, -0.5*p],
        [-0.5*p, -0.5*p, -0.5*p],
    ])

    return ps

def get_square_points(side_len:float):

    p = side_len / 2

    ps = np.array([
        [ p,  p,  p], # +x, +y, +z (0)
        [ p,  p, -p], # +x, +y, -z (1)
        [ p, -p,  p], # +x, -y, +z (2)
        [ p, -p, -p], # +x, -y, -z (3)
        [-p,  p,  p], # -x, +y, +z (4)
        [-p,  p, -p], # -x, +y, -z (5)
        [-p, -p,  p], # -x, -y, +z (6)
        [-p, -p, -p], # -x, -y, -z (7)
    ])

    return ps


def get_square_faces():

    return torch.tensor([
        [0, 4, 5],
        [0, 5, 1],
        [0, 1, 3],
        [0, 2, 3],
        [0, 4, 6],
        [0, 2, 6],
        [7, 6, 4],
        [7, 4, 5],
        [7, 1, 5],
        [7, 1, 3],
        [7, 6, 2],
        [7, 2, 3],
    ]).unsqueeze(0)

if __name__ == '__main__':
    render_shape(shape="square")