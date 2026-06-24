import os
import sys
import traceback

# ==============================================================================
# 1. Path Resolution & Environment Hardening
# ==============================================================================
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"

if os.path.exists(KAGGLE_AGENT_PATH):
    SUBMISSION_DIR = KAGGLE_AGENT_PATH
else:
    SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# ==============================================================================
# 2. Heavy Imports
# ==============================================================================
try:
    import numpy as np
    import torch
    print(f"[*] PyTorch {torch.__version__} loaded successfully.")
    
    from ppo_core import masked_sample, pack_observation
    from orbit_obs_reftrace import OrbitWarsAssembler, schema_metadata
    from config import RunConfig
    
except Exception as e:
    print("\n[CRITICAL] Import phase failed. The agent will likely crash.")
    traceback.print_exc()
    raise

# ==============================================================================
# 3. Global State & Inference Caching
# ==============================================================================
_MODEL = None
_ASSEMBLER = None
_META = None
_DEVICE = torch.device("cpu")

def _extract_state_dict(ckpt: dict) -> dict:
    """Helper to unroll nested checkpoints, identical to eval script."""
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "net", "policy", "actor_critic"):
            v = ckpt.get(k, None)
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                return v
        if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a model state_dict inside the checkpoint.")

def init_agent():
    """Heavy initialization: runs strictly once on Step 0."""
    global _MODEL, _ASSEMBLER, _META

    print("[*] Initializing agent state and loading weights...")
    torch.set_num_threads(2)
    
    _ASSEMBLER = OrbitWarsAssembler(max_steps=500)
    
    cfg = RunConfig.default()
    _MODEL = cfg.make_model()
    _META = _MODEL.meta
    
    model_path = os.path.join(SUBMISSION_DIR, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model weights at {model_path}")
        
    # 1. Load the raw checkpoint
    clean_state_dict = torch.load(model_path, map_location=_DEVICE, weights_only=True)
    _MODEL.load_state_dict(clean_state_dict, strict=True)
    
    _MODEL.eval()
    _MODEL.to(_DEVICE)
    torch.set_grad_enabled(False)

    print("[*] Agent initialized successfully.")

def agent(obs):
    global _MODEL, _ASSEMBLER, _META

    try:
        step = int(obs.get("step_number", obs.get("step", 0)))
        player_id = int(obs.get("player", 0))

        if _MODEL is None or step == 0:
            init_agent()
            _ASSEMBLER.reset(obs, player_id)

        assembled_dict = _ASSEMBLER.assemble(obs, step, compute_expert=False)

        obs_flat = pack_observation(
            global_features=torch.tensor(assembled_dict["global_features"], dtype=torch.float32, device=_DEVICE),
            planet_features=torch.tensor(assembled_dict["planet_features"], dtype=torch.float32, device=_DEVICE),
            edge_features=torch.tensor(assembled_dict["edge_features"], dtype=torch.float32, device=_DEVICE),
            edge_mask=torch.tensor(assembled_dict["edge_mask"], dtype=torch.float32, device=_DEVICE),
            planet_mask=torch.tensor(assembled_dict["planet_mask"], dtype=torch.float32, device=_DEVICE),
            source_mask=torch.tensor(assembled_dict["source_mask"], dtype=torch.float32, device=_DEVICE),
        )

        with torch.no_grad():
            pi_logits, v_logits, v_exp, joint_mask = _MODEL(obs_flat)
            
            B = pi_logits.shape[0]
            P = _MODEL.max_planets
            A = pi_logits.shape[-1]
            F_dim = _META["n_action_options"]
    
            pi_flat = pi_logits.reshape(B * P, A).float()
            jm_flat = joint_mask.reshape(B * P, A).float()
    
            # Greedy sampling
            act_flat, _, _ = masked_sample(pi_flat, jm_flat, greedy=True)
            act_flat = act_flat.reshape(B, P)
    
            t_act = torch.div(act_flat, F_dim, rounding_mode="floor")
            f_act = act_flat % F_dim
            act_pair = torch.stack([t_act, f_act], dim=-1)
            
            act_pair_np = act_pair[0].cpu().numpy().astype(np.int64)

        target_idx_np = act_pair_np[:, 0]
        option_idx_np = act_pair_np[:, 1]

        orders = _ASSEMBLER.map_actions_to_orders(target_idx_np, option_idx_np, obs, step)
        return orders

    except Exception as e:
        print(f"\n[CRASH] Agent crashed at Step {obs.get('step', 'Unknown')}:")
        traceback.print_exc()
        return []