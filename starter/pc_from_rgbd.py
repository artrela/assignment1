import pytorch3d
import numpy as np
from starter.utils import unproject_depth_image, get_points_renderer, get_device
from starter.render_generic import load_rgbd_data
import torch
import tqdm
from PIL import Image, ImageDraw
import imageio
from scipy.spatial.transform.rotation import Rotation

def render_plant():

    device = get_device()
    print(f"{device=}")
    
    data: dict = load_rgbd_data()

    for k,v in data.items():
        if type(v) == np.ndarray:
            data[k] = torch.from_numpy(v).to(device)
        else:
            data[k] = v.to(device)
            
    points1, rgb1 = unproject_depth_image(image=data['rgb1'], mask=data['mask1'], depth=data['depth1'].cpu(), camera=data['cameras1'])
    points2, rgb2 = unproject_depth_image(image=data['rgb2'], mask=data['mask2'], depth=data['depth2'].cpu(), camera=data['cameras2'])

    union_pts =  torch.from_numpy(np.concatenate((points1.cpu(), points2.cpu()), axis=0)).to(device)
    union_rgb =  torch.from_numpy(np.concatenate((rgb1.cpu(), rgb2.cpu()), axis=0)).to(device)
    
    skip = 1
    pc1 = pytorch3d.structures.Pointclouds(points=points1[::skip].unsqueeze(0), features=rgb1[::skip].unsqueeze(0))
    pc2 = pytorch3d.structures.Pointclouds(points=points2[::skip].unsqueeze(0), features=rgb2[::skip].unsqueeze(0))
    pcU = pytorch3d.structures.Pointclouds(points=union_pts[::skip].unsqueeze(0), features=union_rgb[::skip].unsqueeze(0))

    renderer = get_points_renderer(
        image_size=512, background_color=(1, 1, 1),
    )   

    renders = []
    R_180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    for azim in tqdm.tqdm(range(-360, 365, 5)):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=6.0, elev=0, azim=azim, degrees=True)
        R = R @ R_180

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