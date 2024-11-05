import pytorch3d
import numpy as np
from starter.utils import unproject_depth_image, get_points_renderer, get_device
from starter.render_generic import load_rgbd_data
import torch
import tqdm
from PIL import Image, ImageDraw
import imageio
from scipy.spatial.transform.rotation import Rotation

torus = lambda x, y, z, R, r: np.sqrt((np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2)

def sample_torus(n: int, R: float, r: float):

    theta = 2 * np.pi * np.random.rand(n)
    phi = 2 * np.pi * np.random.rand(n)
    
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    points = np.stack((x, y, z), axis=-1)

    return torch.from_numpy(points)
   

def render_torus():

    device = get_device()
    print(f"{device=}")
    

    points = sample_torus(n=50000, R=4, r=2)

    points = points.to(device)
    rgb = torch.ones_like(points) * torch.tensor([[0., 0., 1.]]).to(device)
    rgb = rgb.to(device)

    point_cloud = pytorch3d.structures.Pointclouds(points=points.unsqueeze(0).float(), features=rgb.unsqueeze(0).float())

    renderer = get_points_renderer(
        image_size=512, background_color=(1, 1, 1),
    )   

    renders = []
    for azim in tqdm.tqdm(range(-360, 365, 5)):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=15.0, elev=0, azim=azim, degrees=True)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=60, R=R, T=T, device=device)
        
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)\
        
        renders.append(rend)


    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    
    num_frames = 360 + 45
    duration = num_frames / 60
    imageio.mimsave("results/torus.gif", images, fps=(num_frames / duration))        


if __name__ == "__main__":
    render_torus()