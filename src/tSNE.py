import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MaxPool1d
from openTSNE import TSNE
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
font_path = '/usr/share/fonts/Times/TIMES.TTF'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'Times New Roman'
from transformers import AutoTokenizer, T5Config

class T5Dataset(Dataset):

  def __init__(self, propmt_list, tokenizer,max_length_propmt=1024):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    self.answer_ids = []

    # 设置填充参数为右填充
    tokenizer.padding_side = "right"

    for txt in propmt_list:

      encodings_dict = tokenizer('<s>'+ txt , truncation=True, max_length=max_length_propmt, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx]
  

def emb_convert(file_path, tokenizer):
    data = pd.read_csv(file_path)
    propmt_list = data['prompt'].tolist()
    dataset = T5Dataset(propmt_list,tokenizer,max_length_propmt=140)
    dataloader = DataLoader(dataset, tsne_config['batch_size'], shuffle=False, num_workers=24)
    for step, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(device)
        embeddings = batch.squeeze()
        embeddings = embeddings.reshape(embeddings.shape[0],140).cpu().detach().numpy()
        if step == 0:
            print("shape of embedding:", embeddings.shape)
            embeddings_all = embeddings
        else:
            embeddings_all = np.vstack((embeddings_all, embeddings))

    return  embeddings_all

def plot_tsne(ax, train_tSNE, test_tSNE, train_label, test_label):
    ax.scatter(train_tSNE[:, 0], train_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='lightgrey',
               label=tsne_config[train_label])
    ax.scatter(test_tSNE[:, 0], test_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#44B79C',
               label=tsne_config[test_label])

def main(tsne_config):

    path = "/home/hkqiu/work/PolyGPT/T5/model_save/cls-T5-bs150-bs8-lr5e6-epoch100"
    tokenizer = AutoTokenizer.from_pretrained(path, max_len=tsne_config['blocksize'])
    config = T5Config.from_pretrained(path, output_hidden_states=False)

    train_data = emb_convert(tsne_config['train_path'], tokenizer)
    train_tg = train_data[0:6150, :]
    train_bc = train_data[6150:10390, :]
    train_ae = train_data[10390:15655, :]
    train_hrc = train_data[15655:, :]

    test_data = emb_convert(tsne_config['test_path'], tokenizer)
    test_tg = test_data[0:700, :]
    test_bc = test_data[700:1180, :]
    test_ae = test_data[1180:1765, :]
    test_hrc = test_data[1765:, :]

    val_data = emb_convert(tsne_config['val_path'], tokenizer)

    print("start fitting t-SNE")
    tSNE = TSNE(
        perplexity=tsne_config['perplexity'],
        metric=tsne_config['metric'],
        n_jobs=tsne_config['n_jobs'],
        verbose=tsne_config['verbose'],
    )
    train_tSNE = tSNE.fit(train_data)
    print("finish fitting")

    test_tSNE = train_tSNE.transform(test_data)

    train_tg_tSNE = train_tSNE.transform(train_tg)
    train_bc_tSNE = train_tSNE.transform(train_bc)
    train_ae_tSNE = train_tSNE.transform(train_ae)
    test_tg_tSNE = train_tSNE.transform(test_tg)
    test_bc_tSNE = train_tSNE.transform(test_bc)
    test_ae_tSNE = train_tSNE.transform(test_ae)

    val_tSNE = train_tSNE.transform(val_data)
    
    print("finish t-SNE")

    fig, ax = plt.subplots(figsize=(15, 16))

    # plot_tsne(ax, train_tSNE, test_tSNE, 'train_tSNE', 'test_tSNE')
    # plot_tsne(ax, train_tSNE, train_tg_tSNE, 'train_tSNE', 'train_tg_tSNE')

    # ax.scatter(train_tSNE[:, 0], train_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='gray',
    #            label=tsne_config['train_tSNE'])
    
    ax.scatter(train_tg_tSNE[:, 0], train_tg_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#8CC4F5',
               label=tsne_config['train_tg_tSNE'])

    ax.scatter(test_tg_tSNE[:, 0], test_tg_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#F399C5',
               label=tsne_config['test_tg_tSNE'])

    # ax.scatter(val_tSNE[:, 0], val_tSNE[:, 1], s=50, marker='*', edgecolors='None', linewidths=0.4, c='red',
    #            label=tsne_config['val_tSNE'])
    
    ax.scatter(train_bc_tSNE[:, 0], train_bc_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#BBCEA8',
               label=tsne_config['train_bc_tSNE'])

    ax.scatter(test_bc_tSNE[:, 0], test_bc_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#C3E6A6',
               label=tsne_config['test_bc_tSNE'])

    ax.scatter(train_ae_tSNE[:, 0], train_ae_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#C8BADF',
               label=tsne_config['train_ae_tSNE'])

    ax.scatter(test_ae_tSNE[:, 0], test_ae_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='#F7DF96',
               label=tsne_config['test_ae_tSNE'])
    
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('off')
    ax.legend(fontsize=50, loc='upper center', ncol=3)
    plt.savefig(tsne_config['save_path'], bbox_inches='tight', dpi=1000)


if __name__ == "__main__":

    tsne_config = yaml.load(open("/home/hkqiu/work/PolyGPT/T5/data_code/config_tSNE.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    main(tsne_config)
