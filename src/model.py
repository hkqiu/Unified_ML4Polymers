import torch
import time
import datetime
from transformers import T5Config, T5ForConditionalGeneration, AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.nn.parallel import DataParallel

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class T5Model:
    def __init__(self, config_path, model_path, tokenizer, device=None):
        self.config = T5Config.from_pretrained(config_path, output_hidden_states=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=self.config)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)
    
    def train(self, train_dataloader, validation_dataloader, epochs, 
              learning_rate, warmup_ratio, epsilon, sample_every=1000):
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=epsilon)
        
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps)
        
        training_stats = []
        total_t0 = time.time()
        
        for epoch_i in range(epochs):
            print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
            print("Training...")
            
            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            
            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_masks, b_labels = [b.to(self.device) for b in batch]
                
                self.model.zero_grad()        
                outputs = self.model(b_input_ids, labels=b_labels, 
                                   attention_mask=b_masks)
                
                loss = outputs[0].mean()
                total_train_loss += loss.item()
                
                if step % sample_every == 0 and step != 0:
                    elapsed = format_time(time.time() - t0)
                    print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. '
                          f'Loss: {loss.item():>5,}. Elapsed: {elapsed}.')
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            
            print(f"\n  Average training loss: {avg_train_loss:.2f}")
            print(f"  Training epoch took: {training_time}")
            
            # Validation
            print("\nRunning Validation...")
            t0 = time.time()
            self.model.eval()
            total_eval_loss = 0
            
            for batch in validation_dataloader:
                b_input_ids, b_masks, b_labels = [b.to(self.device) for b in batch]
                
                with torch.no_grad():        
                    outputs = self.model(b_input_ids, 
                                       attention_mask=b_masks,
                                       labels=b_labels)
                    loss = outputs[0].mean()
                    total_eval_loss += loss.item()
            
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)
            
            print(f"  Validation Loss: {avg_val_loss:.2f}")
            print(f"  Validation took: {validation_time}")
            
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        
        print("\nTraining complete!")
        print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")
        
        return training_stats
    
    def save(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def predict(self, prompt, max_length=8, num_return_sequences=1):
        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        sample_outputs = self.model.module.generate(
            generated,
            do_sample=False,
            top_k=100,
            max_length=max_length,
            top_p=0.99,
            num_return_sequences=num_return_sequences
        )
        
        return [self.token.decode(output, skip_special_tokens=False) 
                for output in sample_outputs]