import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
from transformers import get_cosine_schedule_with_warmup
import lightning as pl
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score, recall_score

BACKBONE_NAME = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 4
LABELS = ["real", "fake", "tampered"]

class BayarConv2d(nn.Module):
    """
    A constrained convolution layer that learns to extract high-frequency noise 
    residuals while suppressing image content.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        center = kernel_size // 2
        mask[:, :, center, center] = 0
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        
        weight_sum = masked_weight.sum(dim=(2, 3), keepdim=True)
        norm_weight = masked_weight / (weight_sum + 1e-5)
        
        center = self.kernel_size // 2
        final_weight = norm_weight.clone()
        final_weight[:, :, center, center] = -1.0
        
        return F.conv2d(x, final_weight, stride=1, padding=self.padding)

class NoiseStream(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bayar = BayarConv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten() 
        )
        
    def forward(self, x):        
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = 1
        
        noise_map = self.bayar(x)
        features = self.cnn(noise_map)
        
        if T > 1:
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
            
        return features

class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        self.projector_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.projector_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        v = self.projector_v(x)
        u = self.projector_u(x)
        
        gated_features = v * u
        scores = self.w(gated_features) / self.temperature
        
        attn_weights = F.softmax(scores, dim=1)
        
        weighted_features = torch.sum(x * attn_weights, dim=1)
        
        return weighted_features, attn_weights


class Expert(nn.Module):
    def __init__(self, weights=None, train=True, dim=1024, intermediate_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)

        if weights is not None:
            self.load_state_dict(weights)

        if not train:
            for param in self.parameters():
                param.requires_grad = False 
        else:
            for param in self.parameters():
                param.requires_grad = True


    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MoEJEPALayer(nn.Module):
    def __init__(self, weights=None, num_trainable_experts=3, top_k=3, router_temperature=1.0, hidden_dim=1024, intermediate_dim=4096):
        super(MoEJEPALayer, self).__init__()

        self.num_trainable_experts = num_trainable_experts
        self.top_k = top_k
        self.router_temperature = router_temperature

        self.router = nn.Linear(hidden_dim, self.num_trainable_experts)
        self.shared_expert = Expert(weights=weights, train=False, dim=hidden_dim, intermediate_dim=intermediate_dim)
        self.routed_experts = nn.ModuleList([Expert(weights=weights, train=True, dim=hidden_dim, intermediate_dim=intermediate_dim) for _ in range(self.num_trainable_experts)])

        for expert in self.routed_experts:
            for param in expert.parameters():
                param.data += torch.randn_like(param.data) * 1e-4

        self.current_confidence = None
        self.current_weights = None

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        shared_out = self.shared_expert(x_flat)

        logits = self.router(x_flat)
        logits = logits / self.router_temperature

        experts_prob = torch.sigmoid(logits)

        confidence = experts_prob.max(dim=-1)[0].mean()
        self.current_confidence = confidence
        self.current_weights = experts_prob

        top_k_weights, top_k_indices = torch.topk(experts_prob, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_idx, k_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
                routed_out[token_idx] += weight * expert_out

        final_output = shared_out + routed_out

        return final_output.view(batch_size, seq_len, hidden_dim)

class JointEnergyLoss(nn.Module):
    def __init__(self, m_in=-35.0, m_out=5.0, temperature=1.0, lambda_energy=0.1):
        """
        m_in: The target maximum energy for Real (In-Distribution) images.
        m_out: The target minimum energy for Fake/Tampered (OOD) images.
        temperature: T parameter to scale the logits.
        lambda_energy: How much weight to give the energy loss vs Cross-Entropy.
        """
        super().__init__()
        self.m_in = m_in
        self.m_out = m_out
        self.temperature = temperature
        self.lambda_energy = lambda_energy
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        
        real_mask = (targets == 0)
        fake_mask = (targets > 0)
        
        loss_energy = torch.tensor(0.0, device=logits.device)
        
        if real_mask.any():
            energy_real = energy[real_mask]
            loss_energy += torch.mean(F.relu(energy_real - self.m_in))
            
        if fake_mask.any():
            energy_fake = energy[fake_mask]
            loss_energy += torch.mean(F.relu(self.m_out - energy_fake))
            
        total_loss = loss_ce * (1 - self.lambda_energy) + (self.lambda_energy * loss_energy)
        
        return total_loss, loss_ce, loss_energy.item()
   
class MoEJEPADeepfakeDetector(pl.LightningModule):
    def __init__(self, model_name, num_experts, lr, aux_loss_weight, 
                 router_temperature, weight_decay, train_last_n_layers, lambda_energy,
                 unfreeze_last_n_blocks, pooling_strategy, top_k, do_noise_stream, out_classes):
        super(MoEJEPADeepfakeDetector, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        self.hidden_dim = getattr(self.model.config, 'hidden_size', 1024)
        self.intermediate_dim = getattr(self.model.config, 'intermediate_size', getattr(self.model.config, 'mlp_ratio', 4) * self.hidden_dim)

        self.lr = lr
        self.aux_loss_weight = aux_loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.weight_decay = weight_decay
        self.pooling_strategy = pooling_strategy
        self.do_noise_stream = do_noise_stream
        self.out_classes = out_classes
        self.energy_criterion = JointEnergyLoss(lambda_energy=lambda_energy)
        self.save_hyperparameters()

        print(f"POOLING STRATEGY: {self.pooling_strategy}")
        print(f"DO NOISE STREAM: {self.do_noise_stream}")
        print(f"OUT CLASSES: {self.out_classes}")

        self.fc = nn.Linear(self.hidden_dim, out_classes)
        
        if self.pooling_strategy == "mil":
            self.pooler = GatedAttentionPooling(in_dim=self.hidden_dim, hidden_dim=512)

        if self.do_noise_stream:
            self.noise_stream = NoiseStream()
            self.noise_proj = nn.Linear(128, self.hidden_dim)
            self.ln_semantic = nn.LayerNorm(self.hidden_dim)
            self.ln_noise = nn.LayerNorm(self.hidden_dim)

            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            )
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=out_classes, average="macro")
        
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=out_classes, average="macro")

        self.validation_step_outputs = []

        for param in self.model.parameters():
            param.requires_grad = False

        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        if self.unfreeze_last_n_blocks > 0:
            print(f"Unfreezing the last {self.unfreeze_last_n_blocks} blocks of the encoder...")
            for param in self.model.encoder.layer[-self.unfreeze_last_n_blocks:].parameters():
                param.requires_grad = True

        self.train_last_n_layers = train_last_n_layers
        if self.train_last_n_layers > 0:
            print(f"Adding MoE layers to the last {self.train_last_n_layers} layers of the encoder...")
            self.add_moe_layers(self.train_last_n_layers, num_experts=self.num_experts, router_temperature=self.router_temperature)

    def add_moe_layers(self, train_last_n_layers, num_experts, router_temperature):
        for layer in self.model.encoder.layer[-train_last_n_layers:]:
            weights = layer.mlp.state_dict()
            layer.mlp = MoEJEPALayer(weights=weights, num_trainable_experts=num_experts, top_k=self.top_k, router_temperature=router_temperature, hidden_dim=self.hidden_dim, intermediate_dim=self.intermediate_dim)


    def forward(self, x, labels=None, return_embeddings=False, return_attention=False):
        out = self.model(x)
        full_sequence = out.last_hidden_state

        if self.pooling_strategy == "cls":
            semantic_emb = full_sequence[:, 0, :]
        elif self.pooling_strategy == "mil":
            patch_tokens = full_sequence[:, 1:, :] 
            semantic_emb, attn_weights = self.pooler(patch_tokens)
        else:
            raise ValueError("Invalid pooling strategy")

        if self.do_noise_stream:
            noise_emb_raw = self.noise_stream(x) 

            noise_emb = self.noise_proj(noise_emb_raw) 
            semantic_emb = self.ln_semantic(semantic_emb) 
            noise_emb = self.ln_noise(noise_emb) 

            combined = torch.cat([semantic_emb, noise_emb], dim=1) 
            alpha = self.fusion_gate(combined) 
            
            final_emb = semantic_emb + (alpha * noise_emb) 
        else:
            final_emb = semantic_emb
            noise_emb = None

        preds = self.fc(final_emb)

        if return_attention:
            return preds, attn_weights

        if return_embeddings and self.do_noise_stream:
            return preds, semantic_emb, noise_emb
        
        return preds

    def training_step(self, batch, batch_idx):
        x = batch['pixel_values']
        y = batch['label']

        # Extract embeddings along with predictions during training
        if self.do_noise_stream:
            y_hat, semantic_emb, noise_emb = self(x, labels=y, return_embeddings=True)
        else:
            y_hat = self(x, labels=y)

        energy_total_loss, ce_only_loss, energy_only_loss = self.energy_criterion(y_hat, y)

        aux_loss = 0
        moe_count = 0
        for layer in self.model.encoder.layer[-self.train_last_n_layers:]:
            if isinstance(layer.mlp, MoEJEPALayer):
                probs = layer.mlp.current_weights
                if probs is not None:
                    aux_loss += self.load_balancing_loss(probs)
                    moe_count += 1
        
        if moe_count > 0:
            aux_loss /= moe_count

        total_loss = energy_total_loss + (self.aux_loss_weight * aux_loss)

        preds = torch.argmax(y_hat, dim=1)
        acc = self.train_acc(preds, y)
        f1 = self.train_f1(preds, y)

        self.log('train/loss', total_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/ce_loss', ce_only_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/loss_energy_margin', energy_only_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    
        self.log('train/acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/f1', f1, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        for layer in self.model.encoder.layer[-self.train_last_n_layers:]:
            if isinstance(layer.mlp, MoEJEPALayer):
                layer.mlp.current_weights = None

        return ce_only_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['pixel_values']
        y = batch['label']

        # Ensure labels are valid to prevent CUDA assert error!
        assert y.max() < self.fc.out_features, f"Found label {y.max()} but model only has {self.fc.out_features} classes!"

        y_hat = self(x) 

        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        self.validation_step_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu()
        })

        return loss
    
    def on_validation_epoch_end(self):
        local_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        local_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

        y_true = local_targets.numpy()
        y_pred = local_preds.numpy()

        acc = accuracy_score(y_true, y_pred)
        f1_general = f1_score(y_true, y_pred, average="macro", zero_division=0)

        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)            
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        self.log('val/acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_macro', f1_general, on_epoch=True, prog_bar=True, sync_dist=True)

        class_names = ["real", "fake"] if self.fc.out_features == 2 else ["real", "fake", "tampered"]
        for i, name in enumerate(class_names):
            if i < len(recalls):
                self.log(f'val/acc_{name}', recalls[i], on_epoch=True, sync_dist=True)
                self.log(f'val/f1_{name}', f1_per_class[i], on_epoch=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1) 
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler_config]
    
    def load_balancing_loss(self, router_probs):
        density = router_probs.mean(dim=0)
        num_experts = router_probs.size(-1)
        loss = num_experts * torch.sum(density**2) - 1
        return loss


def load_processor(model_name=BACKBONE_NAME):
    return AutoVideoProcessor.from_pretrained(model_name)


def preprocess_image(img, processor, num_frames=NUM_FRAMES):
    """
    PIL image -> [T, C, H, W] tensor. V-JEPA 2 is a video model, so the image is
    repeated as a static clip of `num_frames` frames (the model was trained with 4).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    pixel_values = processor(img, return_tensors="pt")["pixel_values_videos"]
    return pixel_values.repeat(1, num_frames, 1, 1, 1).squeeze(0)


def load_moejepa(checkpoint_path, model_name=BACKBONE_NAME, device="cpu"):
    """
    Build MoE-JEPA and load trained weights for inference.

    Accepts either the released `.safetensors` file (trained weights only; the frozen
    V-JEPA 2 backbone is downloaded from the Hugging Face Hub) or a full Lightning
    `.ckpt` produced by train/train.py.
    """
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(checkpoint_path, framework="pt") as f:
            hparams = json.loads(f.metadata()["hyper_parameters"])
        state_dict = load_file(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams, state_dict = ckpt["hyper_parameters"], ckpt["state_dict"]

    hparams = {**hparams, "model_name": model_name}
    model = MoEJEPADeepfakeDetector(**hparams)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only weights of the frozen pretrained backbone may be absent from the checkpoint:
    # those are already loaded by AutoModel.from_pretrained().
    pretrained = model.model.state_dict()
    missing = [k for k in missing if not (k.startswith("model.") and k[len("model."):] in pretrained
                                          and ".mlp.routed_experts." not in k and ".mlp.router." not in k)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch.\nMissing: {missing}\nUnexpected: {unexpected}")

    return model.to(device).eval()