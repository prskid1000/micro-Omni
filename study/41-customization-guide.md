# Chapter 41: Customization Guide

[Back to Index](00-INDEX.md)

---

## 🎯 Common Customizations

### 1. Modify Model Size

**Increase capacity**:
```json
// configs/thinker_tiny.json
{
  "n_layers": 6,      // 4 → 6
  "d_model": 384,     // 256 → 384
  "n_heads": 6,       // 4 → 6
  "d_ff": 1536        // 1024 → 1536
}
```

**Decrease for faster training**:
```json
{
  "n_layers": 2,      // 4 → 2
  "d_model": 128,     // 256 → 128
  "n_heads": 2,       // 4 → 2
  "d_ff": 512         // 1024 → 512
}
```

### 2. Change Vocabulary Size

```bash
# Train new tokenizer
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='your_corpus.txt',
    model_prefix='tokenizer',
    vocab_size=10000,  # Increase from 5000
    model_type='bpe'
)
"
```

### 3. Add New Modality

```python
# Example: Add depth sensor data
class DepthEncoder(nn.Module):
    def __init__(self):
        # ... define architecture
        
    def forward(self, depth_map):
        # depth_map: (B, H, W)
        features = self.process(depth_map)
        return features  # (B, T, d)

# Add projector
depth_projector = nn.Linear(depth_dim, thinker_d_model)

# Use in inference
depth_emb = depth_projector(depth_encoder(depth_input))
combined = torch.cat([img_emb, depth_emb, text_emb], dim=1)
```

### 4. Custom Training Data

```python
# data/custom/dataset.py
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': process_image(item['image_path']),
            'audio': process_audio(item['audio_path']),
            'text': item['text'],
            'label': item['label']
        }
```

### 5. Fine-tune on Domain

```bash
# Domain-specific fine-tuning
python sft_omni.py \
  --config configs/omni_sft_tiny.json \
  --data_path data/medical/  # Your domain
  --num_epochs 10 \
  --learning_rate 1e-5  # Lower LR for fine-tuning
```

## 💡 Key Takeaways

✅ **Config files** control model architecture  
✅ **Tokenizer** can be retrained for new vocabulary  
✅ **New modalities** can be added with encoders  
✅ **Custom datasets** easy to integrate  
✅ **Fine-tuning** adapts to specific domains

---

[Back to Index](00-INDEX.md)

