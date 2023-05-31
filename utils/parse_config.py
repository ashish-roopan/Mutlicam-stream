import yaml



def parse_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cam_config = yaml.safe_load(file)
    
    cameras = cam_config['cameras']
    camera_status = cam_config['camera_status']
    ROIs = cam_config['ROIs']
    POIs = cam_config['POIs']
    
    cam_list = []
    for camera, status in zip(cameras, camera_status):
        print(camera, status)
        if status == 'active':
            cam_list.append(camera)

    