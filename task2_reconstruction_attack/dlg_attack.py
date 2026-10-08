"""Deep Leakage from Gradients (DLG) Reconstruction Attack.

References:
- Zhu, L., Liu, Z., Han, S. "Deep Leakage from Gradients." NeurIPS 2019.
- Zhao, B., Mopuri, K. R., Bilen, H. "iDLG: Improved Deep Leakage from Gradients." 2020.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_label_from_gradient(target_gradients):
    """Analytically recover classification label from the last layer bias gradient.
    
    For Cross-Entropy loss with softmax:
        dL / db_i = p_i - y_i
    Since probabilities p_i in (0, 1) and true one-hot y_c = 1:
        dL / db_c = p_c - 1 < 0
        dL / db_j = p_j > 0  (for all j != c)
    Therefore, the ground truth class is argmin(dL / db).
    """
    # The last bias gradient is the final tensor in parameters
    last_bias_grad = target_gradients[-1]
    return int(torch.argmin(last_bias_grad).item())


class DLGAttacker:
    """Implementation of Deep Leakage from Gradients (DLG)."""

    def __init__(
        self,
        model,
        device="cpu",
        optimizer_type="lbfgs",
        lr=1.0,
        max_iter=300,
        cost_fn="l2",
        label_strategy="joint",
        tolerance=1e-6,
        seed=None,
    ):
        """
        Args:
            model: PyTorch model matching the architecture used to generate gradients.
            device: 'cpu' or 'cuda'.
            optimizer_type: 'lbfgs' or 'adam'.
            lr: learning rate for optimizer.
            max_iter: maximum optimization iterations.
            cost_fn: 'l2' (squared Euclidean distance) or 'cosine' (cosine distance).
            label_strategy: 
                - 'joint': optimize dummy label continuous logits alongside dummy image (Zhu et al.).
                - 'analytical': infer label analytically from final bias gradient (Zhao et al. iDLG).
                - 'known': user provides exact label.
            tolerance: early stopping gradient difference threshold.
            seed: random seed for dummy data initialization.
        """
        self.model = copy.deepcopy(model).to(device).eval()
        self.device = device
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.max_iter = max_iter
        self.cost_fn = cost_fn.lower()
        self.label_strategy = label_strategy.lower()
        self.tolerance = tolerance
        self.seed = seed

    def _compute_grad_distance(self, dummy_grads, target_grads):
        """Compute distance between dummy and target gradients."""
        if self.cost_fn == "cosine":
            # Cosine distance across flattened concatenated gradient vectors
            dummy_flat = torch.cat([g.reshape(-1) for g in dummy_grads])
            target_flat = torch.cat([g.reshape(-1) for g in target_grads])
            cos_sim = F.cosine_similarity(dummy_flat.unsqueeze(0), target_flat.unsqueeze(0))
            return 1.0 - cos_sim.squeeze()
        else:
            # Default L2 squared distance (Zhu et al.)
            grad_diff = torch.tensor(0.0, device=self.device)
            for dg, tg in zip(dummy_grads, target_grads):
                grad_diff = grad_diff + ((dg - tg) ** 2).sum()
            return grad_diff

    def attack(
        self,
        target_gradients,
        img_shape=(1, 1, 28, 28),
        known_label=None,
        true_image=None,
        true_label=None,
        log_interval=50,
        progression_steps=None,
    ):
        """Run the DLG reconstruction attack.
        
        Args:
            target_gradients: List of gradient tensors (detached).
            img_shape: Shape tuple of the target image tensor.
            known_label: Optional int label if label_strategy == 'known'.
            true_image: Optional ground-truth image for tracking evaluation metrics (NOT used in optimization).
            true_label: Optional ground-truth label for evaluation.
            log_interval: Frequency of logging iterations.
            progression_steps: Optional list of step numbers at which to capture reconstructed image snapshots.
            
        Returns:
            dict containing:
                - 'reconstructed_image': Tensor[1, 1, 28, 28] clamped to [0, 1]
                - 'reconstructed_label': int predicted label
                - 'loss_history': list of gradient loss values
                - 'progression': dict mapping step -> Tensor snapshot
                - 'iterations': total iterations executed
                - 'converged': boolean flag
                - 'duration': execution time in seconds
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if progression_steps is None:
            progression_steps = [0, 10, 25, 50, 100, 200, 300]

        target_grads = [g.to(self.device).detach() for g in target_gradients]

        # Initialize dummy image with normal distribution
        dummy_data = torch.randn(img_shape, device=self.device, requires_grad=True)

        # Handle label strategy
        optimize_label = False
        fixed_label = None

        if self.label_strategy == "analytical":
            inferred = infer_label_from_gradient(target_grads)
            fixed_label = torch.tensor([inferred], device=self.device, dtype=torch.long)
            recon_label = inferred
        elif self.label_strategy == "known":
            if known_label is None:
                raise ValueError("known_label must be provided when label_strategy='known'")
            fixed_label = torch.tensor([known_label], device=self.device, dtype=torch.long)
            recon_label = int(known_label)
        else:  # 'joint'
            # Jointly optimize dummy label logits
            optimize_label = True
            dummy_label = torch.randn((1, 10), device=self.device, requires_grad=True)

        # Setup optimizer
        if optimize_label:
            params = [dummy_data, dummy_label]
        else:
            params = [dummy_data]

        if self.optimizer_type == "lbfgs":
            optimizer = torch.optim.LBFGS(params, lr=self.lr, max_iter=20)
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        loss_history = []
        progression = {}
        start_time = time.time()
        converged = False

        # Save initial state snapshot
        if 0 in progression_steps:
            progression[0] = torch.clamp(dummy_data.detach().cpu(), 0.0, 1.0)

        for it in range(1, self.max_iter + 1):
            step_loss_val = [0.0]

            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                
                if optimize_label:
                    dummy_loss = F.cross_entropy(dummy_pred, F.softmax(dummy_label, dim=-1))
                else:
                    dummy_loss = F.cross_entropy(dummy_pred, fixed_label)
                    
                dummy_grads = torch.autograd.grad(
                    dummy_loss, list(self.model.parameters()), create_graph=True
                )
                
                grad_diff = self._compute_grad_distance(dummy_grads, target_grads)
                grad_diff.backward()
                step_loss_val[0] = float(grad_diff.item())
                return grad_diff

            current_loss = optimizer.step(closure)
            loss_val = float(current_loss.item()) if current_loss is not None else step_loss_val[0]
            loss_history.append(loss_val)

            # Record progression snapshot
            if it in progression_steps:
                progression[it] = torch.clamp(dummy_data.detach().cpu(), 0.0, 1.0)

            # Logging
            if it % log_interval == 0 or it == 1 or it == self.max_iter:
                log_str = f"Iteration {it:3d}/{self.max_iter}: Grad Distance = {loss_val:.6e}"
                if true_image is not None:
                    curr_img = torch.clamp(dummy_data.detach().cpu(), 0.0, 1.0)
                    mse_now = float(F.mse_loss(curr_img, true_image.detach().cpu()).item())
                    log_str += f" | Current MSE = {mse_now:.6f}"
                print(log_str)

            # Early stopping check
            is_plateau = len(loss_history) >= 5 and abs(loss_history[-1] - loss_history[-2]) < 1e-10
            if loss_val <= self.tolerance or is_plateau:
                converged = True
                progression[it] = torch.clamp(dummy_data.detach().cpu(), 0.0, 1.0)
                break

        duration = time.time() - start_time
        final_image = torch.clamp(dummy_data.detach().cpu(), 0.0, 1.0)

        if optimize_label:
            recon_label = int(torch.argmax(dummy_label.detach().cpu(), dim=-1).item())

        return {
            "reconstructed_image": final_image,
            "reconstructed_label": recon_label,
            "loss_history": loss_history,
            "progression": progression,
            "iterations": it,
            "converged": converged,
            "duration": duration,
        }
