import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')   
from tqdm.auto import tqdm

from baseline.data.custom_datasets import SortDataset
from baseline.models.ctm_sort import ContinuousThoughtMachineSORT
from baseline.tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from baseline.utils.housekeeping import set_seed, zip_python_code
from baseline.utils.losses import sort_loss
from baseline.utils.jepa import add_jepa_args, build_jepa_predictor, compute_jepa_loss, update_jepa_gate_acc
from baseline.tasks.sort.utils import compute_ctc_accuracy, decode_predictions
from baseline.utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from baseline.utils.ctm_model_ideas import add_all_idea_args, ReflexHead
from baseline.utils.ctm_train_ideas import add_train_idea_args, compute_multi_tick_loss, compute_tick_penalty
from baseline.utils.hrm_ideas import add_hrm_idea_args, build_optimizer_from_args, compute_bp_steps, EMATracker, compute_act_q_loss
from baseline.utils.dtt_ideas import add_dtt_args, get_sort_out_dims, per_tick_sort_loss, compute_per_tick_accuracy, compute_per_tick_fine_accuracy, parse_msh_levels, build_msh_synapses, per_tick_sinkhorn_sort_loss, decode_permutation_hungarian, compute_sinkhorn_accuracy

import torchvision
torchvision.disable_beta_transforms_warning()

from autoclip.torch import QuantileClip

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")


warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)

warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Architecture
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input.')
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch.')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--synch_gate_mode', type=str, default='fixed', choices=['fixed', 'soft'],
                        help='Synch subspace selection: fixed=legacy random-pairing indices (su00); soft=learnable softmax gates (su01).')
    parser.add_argument('--synch_gate_temp', type=float, default=1.0,
                        help='Softmax temperature for synch_gate_mode=soft. Lower=peakier (closer to hard indexing).')
    
    parser.add_argument('--iterations', type=int, default=50, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Use deep memory.')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply normalization in NLMs.')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding.', choices=['none', 
                                                                       'learnable-fourier', 
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+',
                        help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile the synapses, backbone, and nlms.')


    # Logging and Saving
    parser.add_argument('--log_dir', type=str, default='logs/scratch',
                        help='Directory for logging.')
    parser.add_argument('--N_to_sort', type=int, default=30, help='N numbers to sort.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Reload only the model from disk?')

    # Tracking
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=2, help='How many minibatches to approx metrics. Set to -1 for full eval')

    # Device
    parser.add_argument('--device', type=int, nargs='+', default=[-1],
                        help='List of GPU(s) to use. Set to -1 to use CPU.')

    add_jepa_args(parser)
    add_all_idea_args(parser)
    add_train_idea_args(parser)
    add_hrm_idea_args(parser)
    add_dtt_args(parser)
    args = parser.parse_args()
    return args




if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()
    # Change the following for sorting
    args.backbone_type = 'none'
    
    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    
    

    

    # Data
    train_data = SortDataset(args.N_to_sort)
    test_data = SortDataset(args.N_to_sort)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1, drop_last=False)
    

    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = get_sort_out_dims(args.N_to_sort, args.sort_loss_mode)

    # For total reproducibility
    # Python 3.x
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)  

    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model sort on {device}')

    # Build model
    model = ContinuousThoughtMachineSORT(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.out_dims-1,  
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,  
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,  
        do_layernorm_nlm=args.do_normalisation,  
        backbone_type='none',
        positional_embedding_type=args.positional_embedding_type,
        out_dims=args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,      
        dropout_nlm=args.dropout_nlm,    
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
        synch_gate_mode=getattr(args, 'synch_gate_mode', 'fixed'),
        synch_gate_temp=getattr(args, 'synch_gate_temp', 1.0),
    ).to(device)

    # --- Setup CTM ideas ---
    nlm_diff = None
    if args.diff_memory:
        mem_lengths = [int(m) for m in args.diff_memory_lengths.split(',')]
        from baseline.utils.ctm_model_ideas import DifferentiatedMemoryNLM
        nlm_diff = DifferentiatedMemoryNLM(
            d_model=args.d_model,
            memory_lengths=mem_lengths,
            hidden_dims_list=[args.memory_hidden_dims] * len(mem_lengths),
            dropout=args.dropout_nlm or args.dropout,
        ).to(device)
        model.nlm_differentiated = nlm_diff

    # Set idea attributes on model
    model.topk_neurons = args.topk_neurons
    model.async_tick_mode = args.async_tick_mode
    model.async_tick_periods = args.async_tick_periods
    model.async_tick_phases = args.async_tick_phases
    model.tick_halt_mode = args.tick_halt_mode
    model.tick_halt_threshold = args.tick_halt_threshold
    model.tick_min_ticks = args.tick_min_ticks
    model.draft_mode = args.draft_mode
    model.draft_revise_weight = args.draft_revise_weight
    model.draft_corrupt_prob = args.draft_corrupt_prob
    model.draft_block_size = args.draft_block_size

    # --- HRM-inspired attributes ---
    model.bp_steps = args.bp_steps
    model.detach_every = args.detach_every

    # --- ACT Q-learning halting ---
    if args.act_halt:
        model.act_halt = True
        model.q_head = torch.nn.Linear(model.synch_representation_size_out, 2).to(device)
        torch.nn.init.zeros_(model.q_head.weight)
        model.q_head.bias.data.fill_(-5.0)
        model.halt_max_steps = min(args.halt_max_steps, args.iterations)
        model.halt_exploration_prob = args.halt_exploration_prob

    # --- Hierarchical recurrence (HRM core) ---
    if args.h_cycles > 1:
        model.h_cycles = args.h_cycles
        model.l_cycles = args.l_cycles if args.l_cycles > 0 else (args.iterations // args.h_cycles)
        import torch.nn as nn
        model.h_synapse = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, args.d_model * 2),
            nn.GLU(),
            nn.LayerNorm(args.d_model),
        ).to(device)

    # --- Multi-Scale Hierarchy (N-level generalization) ---
    msh_levels = parse_msh_levels(getattr(args, 'msh_levels', ''))
    if msh_levels:
        msh_mode = getattr(args, 'msh_mode', 'nested')
        if msh_mode == 'nested':
            levels_product = 1
            for l in msh_levels:
                levels_product *= l
            assert levels_product == args.iterations, \
                f"MSH nested: levels product ({levels_product}) != iterations ({args.iterations})"
            grad_path = msh_levels[-1]
        else:
            from math import gcd
            from functools import reduce
            def _lcm(a, b):
                return a * b // gcd(a, b)
            full_cycle = reduce(_lcm, msh_levels)
            grad_path = f"coprime (full resonance every {full_cycle} steps)"
        print(f"MSH [{msh_mode}]: {len(msh_levels)} levels = {msh_levels}, "
              f"total={args.iterations}, gradient_path={grad_path}")
        model.msh_levels = args.msh_levels
        model.msh_mode = msh_mode
        model.msh_sn_scale = getattr(args, 'msh_sn_scale', 0.0)

        # Build level synapses:
        #   build_msh_synapses always creates len(input_levels) - 1 synapses
        #   nested mode: needs len(msh_levels) - 1 synapses → pass msh_levels as-is
        #   coprime mode: needs len(msh_levels) synapses → pass msh_levels + [1]
        model.msh_synapses = build_msh_synapses(
            msh_levels if msh_mode == 'nested' else msh_levels + [1],
            args.d_model,
            sn_scale=getattr(args, 'msh_sn_scale', 0.0),
            device=device,
        )

        # Learnable gates
        if msh_mode == 'learnable':
            from baseline.utils.dtt_ideas import init_gate_logits
            n_macro_learn = len(msh_levels)
            gate_init = getattr(args, 'msh_gate_init', 'coprime')
            gate_logits = init_gate_logits(
                n_macro_learn, args.iterations, init_mode=gate_init,
                periods=msh_levels,
            ).to(device)
            model.msh_gate_logits = torch.nn.Parameter(gate_logits)
            print(f"  Learnable gates: {n_macro_learn}×{args.iterations}, init={gate_init}")

    # Reflex head
    if args.reflex_head:
        model.reflex_head = ReflexHead(
            synch_size=model.synch_representation_size_out,
            out_dims=args.out_dims,
        ).to(device)
        model.reflex_ticks = args.reflex_ticks
    else:
        model.reflex_head = None

    if hasattr(model, 'synch_representation_size_out'):
        jepa_predictor = build_jepa_predictor(model.synch_representation_size_out, args)
        if jepa_predictor is not None:
            model.cross_tick_predictor = jepa_predictor.to(device)

    
    model.train()

    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)  

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')
    
    

    # Optimizer and scheduler
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip parameters that don't require gradients
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    # Optimizer and scheduler (Common setup)
    if getattr(args, 'optimizer_type', 'adam') == 'adam_atan2':
        optimizer = build_optimizer_from_args(model.parameters(), args)
    elif len(no_decay_names) and args.weight_decay!=0:
        optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay':args.weight_decay},
                                       {'params': no_decay_params, 'weight_decay':0}],
                                  lr=args.lr,
                                  eps=1e-8 if not args.use_amp else 1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    eps=1e-8 if not args.use_amp else 1e-6,
                                    weight_decay=args.weight_decay)
    
    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError
        
   
    
    # Metrics tracking (I like custom)
    # Using batched estimates
    start_iter = 0  # For reloading, keep track of this (pretty tqdm stuff needs it)
    train_losses = []  
    test_losses = []
    train_accuracies = []  # This will be per internal tick, not so simple
    test_accuracies = []
    train_accuracies_full_list = []  # This will be selected according to what is returned by loss function
    test_accuracies_full_list = []
    iters = []

    # Now that everything is initliased, reload if desired
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)
    if args.reload:
        if os.path.isfile(f'{args.log_dir}/checkpoint.pt'):
            print(f'Reloading from: {args.log_dir}/checkpoint.pt')
            checkpoint = torch.load(f'{args.log_dir}/checkpoint.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']
                train_losses = checkpoint['train_losses']
                train_accuracies_full_list = checkpoint['train_accuracies_full_list']
                train_accuracies = checkpoint['train_accuracies']
                test_losses = checkpoint['test_losses']
                test_accuracies_full_list = checkpoint['test_accuracies_full_list']
                test_accuracies = checkpoint['test_accuracies']
                iters = checkpoint['iters']
            else:
                print('Only relading model!')
            if 'torch_rng_state' in checkpoint:
                # Reset seeds, otherwise mid-way training can be obscure (particularly for imagenet)
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if args.do_compile:
        print('Compiling...')
        model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)
        model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)
    
    # --- EMA weight tracker (HRM-Text style) ---
    ema_tracker = None
    if args.ema_decay > 0:
        ema_tracker = EMATracker(model, decay=args.ema_decay)

    # Training
    iterator = iter(trainloader)  # Not training in epochs, but rather iterations. Need to reset this from time to time
    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            # --- BP warmup: dynamically update bp_steps ---
            if args.bp_warmup_ratio > 0:
                model.bp_steps = compute_bp_steps(bi, args.training_iterations,
                                                   args.bp_warmup_ratio,
                                                   args.bp_min_steps, args.bp_max_steps)

            
            
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                if args.do_compile:
                    torch.compiler.cudagraph_mark_step_begin()
                dtt_active = args.sort_loss_mode in ('per_tick_ce', 'per_tick_sinkhorn')
                sinkhorn_active = args.sort_loss_mode == 'per_tick_sinkhorn'
                ideas_active = (args.cross_tick_jepa_weight > 0 or args.tick_halt_mode != 'none' or 
                                args.tick_loss_mode != 'last' or args.reflex_head or
                                args.topk_neurons < 1.0 or args.async_tick_mode != 'none' or
                                args.ema_distill_weight > 0 or args.draft_revise_weight > 0 or
                                args.act_halt or dtt_active)
                if ideas_active:
                    out = model(inputs, return_per_tick_synch=(args.cross_tick_jepa_weight > 0))
                    if isinstance(out[-1], dict):
                        *base, extras = out
                        predictions, certainties, synchronisation = base
                    else:
                        predictions, certainties, synchronisation = out
                        extras = {}
                    if dtt_active:
                        if sinkhorn_active:
                            loss = per_tick_sinkhorn_sort_loss(
                                predictions, targets,
                                certainties=certainties,
                                N_to_sort=args.N_to_sort,
                                progressive_mode=args.dtt_progressive_mode,
                                exp_decay=args.dtt_exp_decay,
                                sinkhorn_iters=args.sinkhorn_iters,
                                sinkhorn_tau=args.sinkhorn_tau,
                                sinkhorn_tau_min=args.sinkhorn_tau_min,
                                anneal=args.sinkhorn_anneal,
                                cur_step=bi,
                                total_steps=args.training_iterations,
                            )
                        else:
                            loss = per_tick_sort_loss(
                                predictions, targets,
                                certainties=certainties,
                                N_to_sort=args.N_to_sort,
                                progressive_mode=args.dtt_progressive_mode,
                                exp_decay=args.dtt_exp_decay,
                                loss_type=args.loss_type,
                            )
                    else:
                        loss = compute_multi_tick_loss(predictions, targets, sort_loss,
                                                       mode=args.tick_loss_mode,
                                                       certainties=certainties,
                                                       weights=args.tick_loss_weights) if args.loss_type == 'softmax_ce' else \
                               compute_multi_tick_loss(predictions, targets,
                                                       lambda p, t: sort_loss(p, t, loss_type=args.loss_type),
                                                       mode=args.tick_loss_mode,
                                                       certainties=certainties,
                                                       weights=args.tick_loss_weights)
                    # JEPA loss
                    if args.cross_tick_jepa_weight > 0 and hasattr(model, 'cross_tick_predictor') and 'synch_per_tick' in extras:
                        from baseline.utils.jepa import compute_jepa_loss
                        synch_per_tick = extras['synch_per_tick']
                        loss = loss + compute_jepa_loss(
                            model.cross_tick_predictor, synch_per_tick,
                            args.cross_tick_jepa_weight, args.cross_tick_jepa_loss,
                            args.cross_tick_jepa_target_stop_grad,
                            main_loss=loss.detach())
                    # Tick compute penalty
                    if args.tick_compute_weight > 0:
                        n_steps = extras.get('n_steps_used', args.iterations)
                        loss = loss + compute_tick_penalty(n_steps, args.iterations, args.tick_compute_weight)
                    # Draft-revise loss
                    if args.draft_revise_weight > 0 and 'draft_prediction' in extras:
                        dp = extras['draft_prediction']
                        dp_flat = dp.reshape(-1, dp.size(-1))
                        tgt_flat = targets.reshape(-1)
                        if dp_flat.size(0) == tgt_flat.size(0):
                            loss = loss + args.draft_revise_weight * F.cross_entropy(dp_flat, tgt_flat.long())
                    # ACT Q-learning loss
                    if args.act_halt and 'q_logits' in extras:
                        from baseline.tasks.sort.utils import decode_predictions
                        decoded = decode_predictions(predictions, blank_label=predictions.size(1)-1)
                        is_correct = torch.tensor(
                            [1.0 if (len(d) == len(t) and torch.equal(d, t.to(d.device))) else 0.0
                             for d, t in zip(decoded, targets)],
                            device=predictions.device)
                        q_T = extras['q_logits'].size(-1)
                        is_correct = is_correct.unsqueeze(1).expand(-1, q_T)
                        loss = loss + compute_act_q_loss(extras['q_logits'], is_correct, weight=args.halt_q_weight)

                    # Gate sparsity loss
                    gate_sparsity_w = getattr(args, 'msh_gate_sparsity', 0.0)
                    if gate_sparsity_w > 0 and hasattr(model, 'msh_gate_logits'):
                        from baseline.utils.dtt_ideas import compute_gate_sparsity_loss
                        loss = loss + gate_sparsity_w * compute_gate_sparsity_loss(model.msh_gate_logits)
                else:
                    out = model(inputs)
                    if isinstance(out[-1], dict):
                        predictions, certainties, synchronisation = out[:-1]
                    else:
                        predictions, certainties, synchronisation = out
                    loss = sort_loss(predictions, targets, loss_type=args.loss_type)
            
                        
            scaler.scale(loss).backward()
        

            if args.gradient_clipping!=-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
            

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # --- EMA weight update ---
            if ema_tracker is not None:
                ema_tracker.update(model)

            if dtt_active:
                if sinkhorn_active:
                    accuracy, _ = compute_sinkhorn_accuracy(predictions, targets, args.N_to_sort)
                else:
                    accuracy = compute_per_tick_accuracy(predictions, targets, args.N_to_sort)
            else:
                accuracy = compute_ctc_accuracy(predictions, targets, predictions.shape[1]-1)
            update_jepa_gate_acc(getattr(model, 'cross_tick_predictor', None), accuracy)
            pbar.set_description(f'Sorting {args.N_to_sort} real numbers. Loss={loss.item():0.3f}. Accuracy={accuracy:0.3f}. LR={current_lr:0.6f}')


            # Metrics tracking and plotting
            if bi%args.track_every==0:# and bi != 0:
                model.eval()
                with torch.inference_mode():
                    

                    inputs, targets = next(iter(testloader))
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pbar.set_description('Tracking: Processing test data')
                    predictions, certainties, synchronisation, pre_activations, post_activations, _, _ = model(inputs, track=True)
                    pbar.set_description('Tracking: Neural dynamics')
                    plot_neural_dynamics(post_activations, min(100, post_activations.shape[-1] // 5 * 5), args.log_dir)

                    imgi = 0


                    
                    ##################################### TRAIN METRICS
                    all_predictions = []
                    all_targets = []
                    all_losses = []
                    
                    iters.append(bi)
                    pbar.set_description('Tracking: Computing loss and accuracy for curves')
                    with torch.inference_mode():
                        loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                out = model(inputs)
                                if isinstance(out[-1], dict):
                                    these_predictions, certainties, synchronisation = out[:-1]
                                else:
                                    these_predictions, certainties, synchronisation = out

                                if dtt_active:
                                    if sinkhorn_active:
                                        loss = per_tick_sinkhorn_sort_loss(these_predictions, targets, N_to_sort=args.N_to_sort, sinkhorn_iters=args.sinkhorn_iters, sinkhorn_tau=args.sinkhorn_tau, sinkhorn_tau_min=args.sinkhorn_tau_min, anneal=args.sinkhorn_anneal)
                                        decoded = decode_permutation_hungarian(these_predictions, N_to_sort=args.N_to_sort)
                                    else:
                                        loss = per_tick_sort_loss(these_predictions, targets, N_to_sort=args.N_to_sort, loss_type=args.loss_type)
                                        decoded = these_predictions.reshape(-1, args.N_to_sort, args.N_to_sort, these_predictions.size(-1))[..., -1].argmax(dim=-1).detach().cpu().numpy()
                                    all_losses.append(loss.item())
                                    all_targets.append(targets.detach().cpu().numpy())
                                    all_predictions.append(decoded)
                                else:
                                    loss = sort_loss(these_predictions, targets)
                                    all_losses.append(loss.item())
                                    all_targets.append(targets.detach().cpu().numpy())
                                    decoded = [d[:targets.shape[1]] for d in decode_predictions(these_predictions, predictions.shape[1]-1)]
                                    decoded = torch.stack([torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device)+targets.shape[1])) if len(d) < targets.shape[1] else d for d in decoded], 0)
                                    all_predictions.append(decoded.detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0 : break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)


                        train_accuracies.append((all_predictions==all_targets).mean())
                        train_accuracies_full_list.append((all_predictions==all_targets).all(-1).mean())
                        train_losses.append(np.mean(all_losses))

                        ##################################### TEST METRICS
                        all_predictions = []
                        all_targets = []
                        all_losses = []
                        loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                out = model(inputs)
                                if isinstance(out[-1], dict):
                                    these_predictions, certainties, synchronisation = out[:-1]
                                else:
                                    these_predictions, certainties, synchronisation = out

                                if dtt_active:
                                    if sinkhorn_active:
                                        loss = per_tick_sinkhorn_sort_loss(these_predictions, targets, N_to_sort=args.N_to_sort, sinkhorn_iters=args.sinkhorn_iters, sinkhorn_tau=args.sinkhorn_tau, sinkhorn_tau_min=args.sinkhorn_tau_min, anneal=args.sinkhorn_anneal)
                                        decoded = decode_permutation_hungarian(these_predictions, N_to_sort=args.N_to_sort)
                                    else:
                                        loss = per_tick_sort_loss(these_predictions, targets, N_to_sort=args.N_to_sort, loss_type=args.loss_type)
                                        decoded = these_predictions.reshape(-1, args.N_to_sort, args.N_to_sort, these_predictions.size(-1))[..., -1].argmax(dim=-1).detach().cpu().numpy()
                                    all_losses.append(loss.item())
                                    all_targets.append(targets.detach().cpu().numpy())
                                    all_predictions.append(decoded)
                                else:
                                    loss = sort_loss(these_predictions, targets)
                                    all_losses.append(loss.item())
                                    all_targets.append(targets.detach().cpu().numpy())
                                    decoded = [d[:targets.shape[1]] for d in decode_predictions(these_predictions, predictions.shape[1]-1)]
                                    decoded = torch.stack([torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device)+targets.shape[1])) if len(d) < targets.shape[1] else d for d in decoded], 0)
                                    all_predictions.append(decoded.detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0 : break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)


                        test_accuracies.append((all_predictions==all_targets).mean())
                        test_accuracies_full_list.append((all_predictions==all_targets).all(-1).mean())
                        test_losses.append(np.mean(all_losses))
                            

                        figacc = plt.figure(figsize=(10, 10))
                        axacc_train = figacc.add_subplot(211)
                        axacc_test = figacc.add_subplot(212)
                        cm = sns.color_palette("viridis", as_cmap=True)
                        axacc_train.plot(iters, train_accuracies, 'b-', alpha=0.7, label='Find grained')   
                        axacc_train.plot(iters, train_accuracies_full_list, 'k--', alpha=0.7, label='Full list')   
                        axacc_test.plot(iters, test_accuracies, 'b-', alpha=0.7, label='Fine grained')        
                        axacc_test.plot(iters, test_accuracies_full_list, 'k--', alpha=0.7, label='Full list')        
                        axacc_train.set_title('Train')
                        axacc_test.set_title('Test')
                        axacc_train.legend(loc='lower right')
                        axacc_train.set_xlim([0, args.training_iterations])
                        axacc_test.set_xlim([0, args.training_iterations])
                        
                        figacc.tight_layout()
                        figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                        plt.close(figacc)

                        figloss = plt.figure(figsize=(10, 5))
                        axloss = figloss.add_subplot(111)
                        axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]}')
                        axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]}')
                        axloss.legend(loc='upper right')

                        axloss.set_xlim([0, args.training_iterations])
                        figloss.tight_layout()
                        figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                        plt.close(figloss)

                model.train()
                            



            # Save model
            if (bi%args.save_every==0 or bi==args.training_iterations-1) and bi != start_iter:
                peak_mem_gb = (torch.cuda.max_memory_allocated(device) / (1024**3)) if "cuda" in device else 0.0
                torch.save(
                    {
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'iteration':bi,
                    'train_accuracies_full_list':train_accuracies_full_list,
                    'train_accuracies':train_accuracies,
                    'test_accuracies_full_list':test_accuracies_full_list,
                    'test_accuracies':test_accuracies,
                    'train_losses':train_losses,
                    'test_losses':test_losses,
                    'iters':iters,
                    'args':args,
                    'peak_memory_gb':peak_mem_gb,
                    'bp_steps':getattr(model,'bp_steps',0),
                    'sort_loss_mode':args.sort_loss_mode,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    } , f'{args.log_dir}/checkpoint.pt')
            
            pbar.update(1)
