from threading import Thread
import numpy as np
import yaml
import cv2, time
from rich import print

class Config_monitor(object):
    def __init__(self, cfg_path='config.yaml'):
        with open(cfg_path, 'r') as file:
            self.cam_config = yaml.safe_load(file)

        self.cfg_path = cfg_path
        self.updated = True # Flag to indicate if config has been updated(initailly set to True to force parse_config() to run

        #. Create thread 
        thread = Thread(target=self.check_for_config_updates, args=())
        thread.daemon = True
        thread.start()

        print('[bold yellow3] Config monitor thread started [/bold yellow3] ', ':smile:', ':thumbs_up:')
        time.sleep(1)   
        
    def parse_config(self):
        cam_config = self.cam_config
        cameras = cam_config['cameras']
        camera_status = cam_config['camera_status']
        ROIs = cam_config['ROIs']
        POIs = cam_config['POIs']
        
        for cam_id, status in camera_status.items():
            if status != 'ON':
                print(f'[bold red] {cam_id} : {status} [/bold red] ', ':skull:')
                del cameras[cam_id]
                del ROIs[cam_id]
                del POIs[cam_id]
        
        self.updated = False
        return cameras, ROIs, POIs

    def check_for_config_updates(self):
        while True:
            with open(self.cfg_path, 'r') as file:
                cam_config = yaml.safe_load(file)
            if cam_config != self.cam_config:
                print('[bold green] Config updated [/bold green] ', ':vampire:')
                self.cam_config = cam_config
                self.updated = True
            
            time.sleep(1)

    
    


