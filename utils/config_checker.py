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
        self.updated = True #. Flag to indicate if config has been updated(initailly set to True to force parse_config() to run

        #. Create thread 
        thread = Thread(target=self.check_for_config_updates, args=())
        thread.daemon = True
        thread.start()
        time.sleep(1)   
        print('\n[bold yellow3] Config monitor thread started [/bold yellow3] \n')
        
    def parse_config(self, return_cfg=False):
        cam_config = self.cam_config
        cameras = cam_config['cameras']
        camera_status = cam_config['camera_status']
        ROIs = cam_config['ROIs']
        POIs = cam_config['POIs']
        models = cam_config['models']
        
        #. Remove cameras that are OFF
        for cam_id, status in camera_status.items():
            if status != 'ON':
                print(f'[bold red] {cam_id} is {status} [/bold red] ')
                del cameras[cam_id]
                del ROIs[cam_id]
                del POIs[cam_id]
                del models[cam_id]
            else:
                print(f'[bold blue] {cam_id} is {status} [/bold blue] \n')

                #. If models has gender detector in it then remove person detector
                if 'gender' in models[cam_id] and 'det' in models[cam_id]:
                    del models[cam_id][models[cam_id].index('det')]
                    print(f'[bold #ff0000] {cam_id}: Removed person detector as gender detector is present [/bold #ff0000] ') 
                    
                #. If model has only track in it add det also
                if 'track' in models[cam_id] and 'det' not in models[cam_id]:
                    models[cam_id].append('det')
                    print(f'[bold #ff0000] {cam_id}: Added person detector as it is required for tracking [/bold #ff0000] ') 
                
        self.updated = False
        if return_cfg:
            return cameras, ROIs, POIs, models, cam_config
        else:
            return cameras, ROIs, POIs, models

    def check_for_config_updates(self):
        while True:
            with open(self.cfg_path, 'r') as file:
                cam_config = yaml.safe_load(file)
            if cam_config != self.cam_config:
                self.cam_config = cam_config
                self.updated = True
            time.sleep(1)

    
    


