"""
Config.py
=========

This file centralizes all hyper-parameters and paths used in our
alpha-aware neural style transfer model based on DualGammaSoftParConv.

The goal of this configuration class is to make **reproduction and ablation**
as transparent as possible: by modifying only a few fields in `Config`, a
reviewer or practitioner can switch between training, testing, and
style-blending modes, change datasets, or adjust the behavior of the
DualGammaSoftParConv module.

Key usage
---------

1. Phase control
   - `phase = 'train'`:
       Default mode for training the model from scratch or resuming from a
       checkpoint.
   - `phase = 'test'`:
       Runs the model in inference mode on predefined content / style folders.
       In this mode, some paths and parameters are overridden by the block
       inside `if phase == 'test':`.
   - `phase = 'style_blending'` (if used in your local code):
       Optional mode for style-mixing or visualization experiments.
   Please ensure that `phase` is set correctly before running `train.py`
   or `test.py`.

2. Experiment naming and output organization
   - `file_n`:
       String identifier for a specific experiment run, typically encoding the
       method name and date (e.g., `DualGammaSoftParConv-2025-11-16-V10`).
       This name is used to construct:
         * `ckpt_dir`: directory for saving / loading checkpoints.
         * `img_dir`: directory for saving generated images.
       When starting a **new training run** or testing a **different checkpoint**,
       please update `file_n` accordingly.

3. Dataset paths
   - `content_dir`, `style_dir`:
       Root folders for content and style images.
       In training mode they typically point to large-scale datasets
       (e.g., COCO as content, WikiArt-RGBA as style).
       In test mode they can be redirected to a smaller, curated evaluation set
       for qualitative and quantitative analysis.

4. DualGammaSoftParConv and Octave Convolution parameters
   - `alpha_in`, `alpha_out`, `freq_ratio`:
       Control the ratio of low-/high-frequency channels when using octave
       convolutions in the encoder. (from AesFA)
   - `gamma_in`, `gamma_out`, `k_in`, `k_out`, `p_in`, `p_out`:
       Hyper-parameters of the DualGammaSoftParConv module. They regulate how
       the alpha channel modulates feature responses in visible vs. transparent
       regions (e.g., power-law scaling strength and stability of the alpha
       transformation). These values were selected empirically and kept fixed
       across our main experiments unless otherwise stated. (from DGSoftPartConv)

5. Optimization and preprocessing
   - `n_iter`, `batch_size`, `lr`, `lr_policy`, `lr_decay_iters`, `beta1`:
       Standard training hyper-parameters (number of iterations, learning rate,
       scheduler, and optimizer settings).
   - `load_size`, `crop_size`:
       Image resizing and random crop size used during training.

6. Loss weights
   - `lambda_percept`, `lambda_perc_cont`, `lambda_perc_style`, `lambda_const_style`:
       Scalar weights for different terms in the total objective (perceptual
       loss, content alignment, style alignment, and style consistency). These
       were tuned once and then fixed for all reported experiments.

7. Reproducibility
   - All experiments reported in the paper can be reproduced by:
       (i) setting `phase = 'train'`,  
       (ii) choosing the appropriate `content_dir` and `style_dir`,  
       (iii) training until convergence (`n_iter`), and  
       (iv) switching to `phase = 'test'` with the corresponding `file_n`
           and checkpoint index (`ckpt_iter`, `ckpt_epoch`).
   - For clarity, we avoid any hidden configuration outside this class; all
     essential experimental settings are exposed here.
"""


class Config:
    # -------------------------------------------------------------------------
    # Phase and training mode
    # -------------------------------------------------------------------------
    phase = 'train'         # Must be 'train', 'test', or (optionally) 'style_blending'
    train_continue = 'off'  # 'on' to resume from an existing checkpoint, 'off' to start from scratch

    # -------------------------------------------------------------------------
    # Dataset settings (default for training)
    # -------------------------------------------------------------------------
    data_num = 60000        # Maximum number of training pairs (content-style combinations)

    content_dir = './train2017'        # Root directory for content images (e.g., COCO)
    style_dir = './wikiart_rgba_out'   # Root directory for style images with alpha channel
    cuda_device = 'cuda:0'             # CUDA device identifier (e.g., 'cuda:0', 'cuda:1')

    # -------------------------------------------------------------------------
    # Experiment naming and paths
    # -------------------------------------------------------------------------
    file_n = 'DualGammaSoftParConv-2025-11-16-V10'  # Experiment name: update for each new training or test run
    ckpt_dir = './ckpt/' + file_n                   # Directory to save/load checkpoints
    img_dir = './Generated_images/' + file_n        # Directory to save generated images

    # -------------------------------------------------------------------------
    # Test-time overrides
    # -------------------------------------------------------------------------
    if phase == 'test':
        multi_to_multi = True      # If True, allow many-to-many content-style pairing at test time
        test_content_size = 256    # Test-time content image size (after resizing)
        test_style_size = 256      # Test-time style image size (after resizing)

        # Test-time content and style directories
        content_dir = './input/contents/alpha'
        style_dir = './input/styles/wikiart/stylealpha_contrast'

        # Checkpoint to be loaded at test time (chosen by iteration and epoch index)
        ckpt_iter = 160000 * 10
        ckpt_epoch = 22
        ckpt_name = 'model_iter_' + str(ckpt_iter) + '_epoch_' + str(ckpt_epoch) + '.pth'

        # Test-time output directory
        img_dir = './partialAesFA'

    # -------------------------------------------------------------------------
    # VGG backbone for perceptual loss
    # -------------------------------------------------------------------------
    vgg_model = './vgg_normalised.pth'   # Pre-trained, normalized VGG-19 model

    # -------------------------------------------------------------------------
    # Basic optimization parameters
    # -------------------------------------------------------------------------
    n_iter = 160000 * 5   # Total number of training iterations
    batch_size = 12       # Training batch size
    lr = 0.0001           # Initial learning rate
    lr_policy = 'step'    # Learning rate policy (e.g., 'step' schedule)
    lr_decay_iters = 50   # Step size (in epochs) for learning rate decay
    beta1 = 0.0           # β1 parameter of the Adam optimizer

    # -------------------------------------------------------------------------
    # Pre-processing parameters
    # -------------------------------------------------------------------------
    load_size = 512       # Images are first resized to this size (shorter side)
    crop_size = 256       # Random crop size used during training

    # -------------------------------------------------------------------------
    # Model architecture parameters
    # -------------------------------------------------------------------------
    input_nc = 3          # Number of input channels (RGB)
    nf = 64               # Number of feature channels after the first encoder layer
    output_nc = 3         # Number of output channels (RGB)
    style_kernel = 3      # Kernel size used in style modulation layers

    # -------------------------------------------------------------------------
    # Octave Convolution parameters (high/low frequency split)
    # -------------------------------------------------------------------------
    alpha_in = 0.5        # Ratio of low-frequency channels in the input
    alpha_out = 0.5       # Ratio of low-frequency channels in the output
    freq_ratio = [1, 1]   # Relative weights [high, low] at the last octave layer

    # -------------------------------------------------------------------------
    # Loss weighting
    # -------------------------------------------------------------------------
    lambda_percept = 1.0        # Global weight for perceptual loss
    lambda_perc_cont = 3.0      # Weight for content-related perceptual term
    lambda_perc_style = 10.0    # Weight for style-related perceptual term
    lambda_const_style = 5.0    # Weight for style consistency / regularization term

    # -------------------------------------------------------------------------
    # Miscellaneous training options
    # -------------------------------------------------------------------------
    norm = 'instance'           # Type of normalization layer (e.g., instance normalization)
    init_type = 'normal'        # Weight initialization method (e.g., normal distribution)
    init_gain = 0.02            # Standard deviation for 'normal' initialization
    no_dropout = 'store_true'   # Flag used by argparse; if set, disables dropout layers
    num_workers = 4             # Number of workers for PyTorch DataLoader

    # -------------------------------------------------------------------------
    # DualGammaSoftParConv hyper-parameters
    # -------------------------------------------------------------------------
    gamma_in = 0.9              # Gamma exponent applied inside visible regions (alpha-guided)
    gamma_out = 1.1             # Gamma exponent applied outside / near transparent regions

    k_in = 1.5                  # Scaling factor for feature modulation inside visible regions
    k_out = 1.2                 # Scaling factor for feature modulation outside / near transparent regions

    p_in = 1.0                  # Additional power / stabilization term for inside-region alpha transform
    p_out = 1.0                 # Additional power / stabilization term for outside-region alpha transform