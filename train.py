import configparser
from rmi.data.lafan1_dataset import LAFAN1Dataset
from torch.utils.data import DataLoader

def train():
    config = configparser.ConfigParser()
    config.read('config/config_base.ini')
    data_dir = config['data']['data_dir']
    batch_size = int(config['data']['batch_size'])
    
    lafan_dataset = LAFAN1Dataset(lafan_path=data_dir)
    lafan_data_loader =  DataLoader(lafan_dataset, batch_size=batch_size, shuffle=True)
    lafan_data_loader


if __name__ == '__main__':
    train()
