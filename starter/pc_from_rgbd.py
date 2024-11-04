import pytorch3d
import numpy as np
from starter.utils import unproject_depth_image, get_points_renderer, get_device
from starter.render_generic import load_rgbd_data
import torch
import tqdm
from PIL import Image, ImageDraw
import imageio

def render_plant():

    device = get_device()
    print(f"{device=}")
    
    data: dict = load_rgbd_data()

    for k,v in data.items():
        if type(v) == np.ndarray:
            data[k] = torch.from_numpy(v).to(device)
        else:
            data[k] = v.to(device)
            
    points1, rgb1 = unproject_depth_image(image=data['rgb1'], mask=data['mask1'], depth=data['depth1'], camera=data['cameras1'])
    points2, rgb2 = unproject_depth_image(image=data['rgb2'], mask=data['mask2'], depth=data['depth2'], camera=data['cameras2'])

    union_pts =  torch.from_numpy(np.concatenate((points1, points2), axis=0)).to(device)
    union_rgb =  torch.from_numpy(np.concatenate((rgb1, rgb2), axis=0)).to(device)
    
    skip = 2
    pc1 = pytorch3d.structures.Pointclouds(points=points1[::skip].unsqueeze(0), features=rgb1[::skip].unsqueeze(0))
    pc2 = pytorch3d.structures.Pointclouds(points=points2[::skip].unsqueeze(0), features=rgb2[::skip].unsqueeze(0))
    pcU = pytorch3d.structures.Pointclouds(points=union_pts[::skip].unsqueeze(0), features=union_rgb[::skip].unsqueeze(0))

    renderer = get_points_renderer(
        image_size=256, background_color=(1, 1, 1),
    )   

    renders = []
    for azim in tqdm.tqdm(range(-60, 65, 5)):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=6.0, azim=azim, degrees=True)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=60, R=R, T=T, device=device)
        
        tqdm.tqdm.write(f"=======================================")
        pc = []
        for i, point_cloud in enumerate((pc1, pc2, pcU)):
            tqdm.tqdm.write(f"Completed Point Cloud {i} at Azimuth {azim}")
            rend = renderer(point_cloud, cameras=cameras)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)\
            pc.append(rend)
        
        renders.append(np.hstack(pc))


    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    
    num_frames = 360 + 45
    duration = num_frames / 60
    imageio.mimsave("results/plant_pc.gif", images, fps=(num_frames / duration))        


if __name__ == "__main__":
    render_plant()