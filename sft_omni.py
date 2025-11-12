
import argparse, json, os, torch, csv, json as js
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio

from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny

class MixDataset(Dataset):
    def __init__(self, text_path, image_manifest, image_root, asr_csv, ctx=1024):
        self.text = [l.strip() for l in open(text_path, 'r', encoding='utf-8') if l.strip()]
        self.images = js.load(open(image_manifest, 'r', encoding='utf-8'))
        self.image_root = image_root
        self.asr = [r for r in csv.DictReader(open(asr_csv, 'r', encoding='utf-8'))]
        self.tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        self.ctx = ctx

    def __len__(self): return max(len(self.text), len(self.images), len(self.asr))

    def __getitem__(self, i):
        it = {}
        if i < len(self.text):
            it["text"] = self.text[i]
        else:
            it["text"] = "Describe the image or audio."
        if i < len(self.images):
            img_path = os.path.join(self.image_root, self.images[i]["image"])
            it["image"] = img_path; it["caption"] = self.images[i]["caption"]
        if i < len(self.asr):
            it["audio"] = self.asr[i]["wav"]; it["trans"] = self.asr[i]["text"]
        return it

def mix_collate_fn(batch):
    """Custom collate function that handles missing keys"""
    result = {}
    # Get all possible keys from the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # For each key, collect values (use None for missing)
    for key in all_keys:
        values = []
        for item in batch:
            if key in item:
                values.append(item[key])
            else:
                values.append(None)
        result[key] = values if None not in values else values
    return result

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ds = MixDataset(cfg["sft_mix"]["text_path"], cfg["sft_mix"]["image_manifest"], cfg["sft_mix"]["image_root"], cfg["sft_mix"]["asr_csv"], cfg["ctx_len"])
    print(f"Dataset size: {len(ds)}")
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 2), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=mix_collate_fn)
    print(f"DataLoader created, starting training...")

    # load components
    thinker_cfg = cfg.get("thinker", {})
    think = ThinkerLM(
        thinker_cfg.get("vocab_size", 5000),
        thinker_cfg.get("n_layers", 4),
        thinker_cfg.get("d_model", 256),
        thinker_cfg.get("n_heads", 4),
        thinker_cfg.get("d_ff", 1024),
        thinker_cfg.get("dropout", 0.1),
        thinker_cfg.get("rope_theta", 10000),
        cfg["ctx_len"]
    ).to(device)
    if os.path.exists(os.path.join(cfg["thinker_ckpt"], "thinker.pt")):
        think.load_state_dict(torch.load(os.path.join(cfg["thinker_ckpt"], "thinker.pt"), map_location=device))

    # Load audio encoder config from checkpoint or use defaults
    audio_cfg_path = "configs/audio_enc_tiny.json"
    if os.path.exists(audio_cfg_path):
        audio_cfg = json.load(open(audio_cfg_path))
        aud = AudioEncoderTiny(
            d=audio_cfg.get("d_model", 192),
            heads=audio_cfg.get("n_heads", 3),
            ff=audio_cfg.get("d_ff", 768),
            layers=audio_cfg.get("n_layers", 4),
            dropout=audio_cfg.get("dropout", 0.1)
        ).to(device)
    else:
        aud = AudioEncoderTiny().to(device)
    if os.path.exists(os.path.join(cfg["audio_ckpt"], "audio_enc.pt")):
        aud.load_state_dict(torch.load(os.path.join(cfg["audio_ckpt"], "audio_enc.pt"), map_location=device)["enc"])

    # Load vision encoder config from checkpoint or use defaults
    vision_cfg_path = "configs/vision_tiny.json"
    if os.path.exists(vision_cfg_path):
        vision_cfg = json.load(open(vision_cfg_path))
        vis = ViTTiny(
            img_size=vision_cfg.get("img_size", 224),
            patch=vision_cfg.get("patch", 16),
            d=vision_cfg.get("d_model", 128),
            layers=vision_cfg.get("n_layers", 4),
            heads=vision_cfg.get("n_heads", 2),
            ff=vision_cfg.get("d_ff", 512),
            dropout=vision_cfg.get("dropout", 0.1)
        ).to(device)
    else:
        vis = ViTTiny().to(device)
    if os.path.exists(os.path.join(cfg["vision_ckpt"], "vision.pt")):
        vis.load_state_dict(torch.load(os.path.join(cfg["vision_ckpt"], "vision.pt"), map_location=device)["vit"])

    # simple projectors - use actual model dimensions
    audio_dim = audio_cfg.get("d_model", 192) if os.path.exists(audio_cfg_path) else 384
    vision_dim = vision_cfg.get("d_model", 128) if os.path.exists(vision_cfg_path) else 192
    thinker_d_model = thinker_cfg.get("d_model", 256)
    proj_a = torch.nn.Linear(audio_dim, thinker_d_model).to(device)
    proj_v = torch.nn.Linear(vision_dim, thinker_d_model).to(device)
    opt = torch.optim.AdamW(list(think.parameters())+list(proj_a.parameters())+list(proj_v.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, win_length=400, n_mels=128).to(device)
    tok_model = os.path.join(cfg["thinker_ckpt"], "tokenizer.model")
    from omni.tokenizer import BPETokenizer
    tok = BPETokenizer(tok_model)

    def pack_text(prompt, answer, ctx):
        ids = [1] + tok.encode(prompt + " " + answer)
        ids = ids[:ctx]
        x = torch.tensor(ids + [0]*(ctx-len(ids)), dtype=torch.long)
        y = x.clone(); y[:-1]=x[1:]; y[-1]=0
        return x,y

    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 20)
    prompt = cfg.get("prompt", "You are an omni assistant.")
    
    print(f"Starting training: max_epochs={max_epochs}, max_steps={cfg['max_steps']}, batch_size={cfg.get('batch_size', 2)}")
    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        for step,data in enumerate(dl):
            # build a simple batch mixing modalities by concatenating features into the text stream (conceptual)
            B = len(data["text"])
            loss_acc = 0.0
            for b in range(B):
                parts = []
                # image
                if "image" in data and isinstance(data["image"], list) and data["image"][b] is not None:
                    img_path = data["image"][b]
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        img = transforms.Resize((224,224))(img); img = transforms.ToTensor()(img).to(device).unsqueeze(0)
                        cls,_ = vis(img)
                        a = proj_v(cls)  # (1,1,256)
                        # project to fake tokens (repeat)
                        # (for simplicity we ignore true fusion and just supervise caption generation)
                # audio
                if "audio" in data and isinstance(data["audio"], list) and data["audio"][b] is not None:
                    audio_path = data["audio"][b]
                    if audio_path and os.path.exists(audio_path):
                        wav,_ = torchaudio.load(audio_path); wav = wav.to(device)
                        mel = mel_spec(wav)[0].T.unsqueeze(0)
                        a_feat = proj_a(aud(mel))  # (1,T,256)

                ans = data["text"][b]
                x,y = pack_text(prompt, ans, cfg["ctx_len"])
                x,y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
                logits = think(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_acc += float(loss)

            if step % print_freq == 0:
                print("step", step, "loss", loss_acc/B)
            if step >= cfg["max_steps"]:
                os.makedirs(cfg["save_dir"], exist_ok=True)
                torch.save({"thinker": think.state_dict(), "proj_a": proj_a.state_dict(), "proj_v": proj_v.state_dict()}, os.path.join(cfg["save_dir"], "omni.pt"))
                print("Saved", cfg["save_dir"])
                return
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            os.makedirs(cfg["save_dir"], exist_ok=True)
            torch.save({"thinker": think.state_dict(), "proj_a": proj_a.state_dict(), "proj_v": proj_v.state_dict()}, os.path.join(cfg["save_dir"], "omni.pt"))
            print("Saved", cfg["save_dir"], "at end of epoch", epoch, "step", step)
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
