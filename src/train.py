import os
import yaml
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from .dataset import GPT2Dataset
from .model import T5Model

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(tokenizer, config):
    df_train = pd.read_csv(config['data']['train_path'])
    df_train.dropna(inplace=True)
    prompt_list_train = df_train['prompt'].tolist()
    answer_list_train = df_train['target'].tolist()
    
    df_test = pd.read_csv(config['data']['test_path'])
    df_test.dropna(inplace=True)
    prompt_list_test = df_test['prompt'].tolist()
    answer_list_test = df_test['target'].tolist()
    
    return prompt_list_train, answer_list_train, prompt_list_test, answer_list_test

def main():
    config = load_config('configs/default.yaml')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['token_path'],
        bos_token='<s>',
        eos_token='</s>',
        pad_token='[PAD]',
        padding_side='right'
    )
    
    # Add custom tokens
    custom_tokens = [
        'Predict the atomization energy of the following SMILES:',
        'Predict the bandgap crystal of the following SMILES:',
        'Predict the Tg of the following SMILES:',
        'Predict the heat resistance class of the following SMILES:'
    ]
    tokenizer.add_tokens(custom_tokens)
    
    # Prepare data
    train_prompts, train_answers, test_prompts, test_answers = prepare_data(tokenizer, config)
    
    # Create datasets
    train_dataset = GPT2Dataset(
        train_prompts, train_answers, tokenizer,
        max_length_propmt=config['data']['max_length_prompt'],
        max_length_answer=config['data']['max_length_answer']
    )
    
    val_dataset = GPT2Dataset(
        test_prompts, test_answers, tokenizer,
        max_length_propmt=config['data']['max_length_prompt'],
        max_length_answer=config['data']['max_length_answer']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config['training']['batch_size']
    )
    
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = T5Model(
        config_path=config['model']['config_path'],
        model_path=config['model']['model_path'],
        tokenizer=tokenizer
    )
    
    # Train model
    training_stats = model.train(
        train_dataloader,
        validation_dataloader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        warmup_ratio=config['training']['warmup_ratio'],
        epsilon=config['training']['epsilon']
    )
    
    # Save model
    if not os.path.exists(config['output']['dir']):
        os.makedirs(config['output']['dir'])
    
    model.save(config['output']['dir'])
    print(f"Model saved to {config['output']['dir']}")

if __name__ == "__main__":
    main()