import json
import os
import platform
import time

import numpy as np

import sshtunnel
from pymongo import MongoClient
# from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.util.config import cfg


class Reporter:

    def __init__(self):
        """
        Greps json data which is stored in the cfg[meas_data_folder] and sends it to mongoDB
        on cyberdyne (lea38) via sshtunnel on port MONGODB_PORT
        """

        MONGODB_PORT = cfg['MONGODB_PORT']

        node = platform.uname().node

        if node in cfg['lea_vpn_nodes']:
            self.server_name = 'lea38'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT)}
            self.save_folder = './' + cfg['meas_data_folder']
        else:
            # assume we are on a node of pc2 -> connect to frontend and put data on prt 12001
            # from there they can be grep via permanent tunnel from cyberdyne
            self.server_name = 'fe.pc2.uni-paderborn.de'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT),
                            'ssh_username': 'webbah'}

            self.save_folder = '/scratch/hpc-prf-reinfl/weber/OMG/' + cfg['meas_data_folder']

    def save_to_mongodb(self, database_name: str, col: str = ' trails', data=None):
        """
        Stores data to database in document col
        """
        with sshtunnel.open_tunnel(self.server_name, **self.tun_cfg) as tun:
            with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
                db = client[database_name]
                trial_coll = db[col]  # get collection named col
                trial_coll.insert_one(data)

    def oldest_file_in_tree(self, extension=".json"):
        """
        Returns the oldest file-path string
        """
        print(os.getcwd())
        return min(
            (os.path.join(dirname, filename)
             for dirname, dirnames, filenames in os.walk(self.save_folder)
             for filename in filenames
             if filename.endswith(extension)),
            key=lambda fn: os.stat(fn).st_mtime)

    def json_to_mongo_via_sshtunnel(self):

        if not len(os.listdir(self.save_folder)) == 0:

            try:
                oldest_file_path = self.oldest_file_in_tree()
            except(ValueError) as e:
                print('Folder seems empty or no matching data found!')
                print(f'ValueError{e}')
                print('Empty directory! Go to sleep for 5 minutes!')
                time.sleep(5 * 60)
                return

            with open(oldest_file_path) as json_file:
                data = json.load(json_file)

            successfull = False
            retry_counter = 0

            while not successfull:
                try:
                    now = time.time()
                    if os.stat(oldest_file_path).st_mtime < now - 60:
                        self.save_to_mongodb(database_name=data['Database name'],
                                             col='Trial_number_' + data['Trial number'], data=data)
                        print('Reporter: Data stored successfully to MongoDB and will be removed locally!')
                        os.remove(oldest_file_path)
                        successfull = True
                except (sshtunnel.BaseSSHTunnelForwarderError) as e:
                    wait_time = np.random.randint(1, 60)
                    retry_counter += 1
                    if retry_counter > 10:
                        print('Stopped after 10 connection attempts!')
                        raise e
                    print(f'Reporter: Could not connect via ssh to frontend, retry in {wait_time} s')
                    time.sleep(wait_time)

        else:
            print('Empty directory! Go to sleep for 5 minutes!')
            time.sleep(5 * 60)


if __name__ == "__main__":

    reporter = Reporter()
    print("Starting Reporter for logging from local savefolder to mongoDB")
    # print(reporter.oldest_file_in_tree())
    while True:
        reporter.json_to_mongo_via_sshtunnel()
