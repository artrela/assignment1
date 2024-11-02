import tqdm
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
from starter.utils import get_mesh_renderer, get_device
from starter.render_mesh import load_cow_mesh   
import pytorch3d
import numpy as np
from PIL import Image, ImageDraw
import imageio

def retexture_mesh(cow_path="data/cow.obj"):
    device = get_device()

    renderer = get_mesh_renderer(device=device)
    
    color1, color2 = torch.tensor([[1,0,0]]).T, torch.tensor([[0,1,0]]).T

       # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = get_colors(color1, color2, vertices)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    num_frames = 360 + 45
    duration = num_frames / 60
    elevations = torch.arange(45)
    azimuths = torch.arange(360)

    renders = []
    for elev in tqdm.tqdm(elevations):
        
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3.0, elev=elev, azim=0, degrees=True)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=60, R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)
    
    for azim in tqdm.tqdm(azimuths):
        
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3.0, elev=elevations[-1], azim=azim, degrees=True)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=60, R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)
    
        

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)

        elev_idx = min(44, i)
        azim_idx = i - 45 if i > 45 else 0
        
        draw.text((20, 20), f"azimuth: {azimuths[azim_idx]+1}, elevation: {elevations[elev_idx]+1}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimsave("results/change_color.gif", images, fps=(num_frames / duration))


def get_colors(color1, color2, points):
    '''
    points.shape = 
    '''

    z = points[0, :, -1]

    z_min, z_max = torch.min(z), torch.max(z)

    alpha = (z - z_min) / (z_max - z_min)
    color = alpha * color2 + (1 - alpha) * color1

    return color.T.unsqueeze(0)
    



if __name__ == "__main__":
    retexture_mesh()