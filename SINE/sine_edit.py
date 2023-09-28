import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext


sys.path.append("./SINE")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler




def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def set_args(log_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="SINE/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./saved_models/sd-v1-4-full-ema.ckpt",
        help="path to checkpoint of model",
    )   
    
    parser.add_argument(
        "--sin_config",
        type=str,
        default="SINE/configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--sin_ckpt",
        type=str,
        required=True,
    )
     
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--img_save_dir",
        type=str,
        required=False,
    )
    parser.add_argument("--single_guidance", action="store_true")
    parser.add_argument("--range_t_max", type=int, default=400)
    parser.add_argument("--range_t_min", type=int, default=1)
    parser.add_argument("--cond_beta", type=float, default=0.5)

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    opt = parser.parse_args(args=[
        "--ddim_eta", "0.0",
        "--n_iter", "1",
        "--scale", "10.0",
        "--ddim_steps", "100",
        "--sin_ckpt", log_dir + "/checkpoints/last.ckpt",
        # "--prompt", prompt,
        # "--cond_beta", str(v),
        # "--range_t_min",  str(1000 - K),
        "--range_t_max",  "1000", 
        "--single_guidance",
        "--H", "512", "--W", "512",
        "--n_samples", "1",
        # "--outdir", log_dir, 
        # "--img_save_dir", img_save_dir,
        # "--seed", str(seed)
    ])
    
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # model = instantiate_from_config(config.model)
    # opt.cond_beta_sin = 1. - opt.cond_beta
    # model.extra_config = vars(opt)
    # model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    
    sin_config = OmegaConf.load(f"{opt.sin_config}")
    sin_model = load_model_from_config(sin_config, f"{opt.sin_ckpt}")
    sin_model = sin_model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        if opt.single_guidance:
            from ldm.models.diffusion.guidance_ddim import DDIMSinSampler
            sampler = DDIMSinSampler(model, sin_model)
        else:
            sampler = DDIMSampler(model)

    return model, sin_model, sampler, opt



def sine_edit(model, sin_model, sampler, opt,
    log_dir, prompt, img_save_dir, v, K, seed):
    
    # set opt parameters
    opt.outdir = log_dir
    opt.prompt = prompt
    opt.img_save_dir = img_save_dir
    opt.cond_beta = v
    opt.range_t_min = 1000 - K
    opt.seed = seed

    seed_everything(opt.seed)

    opt.cond_beta_sin = 1. - opt.cond_beta
    model.extra_config = vars(opt)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                with sin_model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            
                            
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                                uc_sin = sin_model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                                
                            if opt.single_guidance:
                                b = len(prompts)
                                prompt = prompts[0]
                                prompts = [prompt.split('[SEP]')[0].strip()] * b
                                prompts_single = [prompt.split('[SEP]')[1].strip()] * b
                            
                            c = model.get_learned_conditioning(prompts)
                            c_sin = sin_model.get_learned_conditioning(prompts_single)
                            
                            
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            conditioning_single=c_sin,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            unconditional_conditioning_single=uc_sin,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(opt.img_save_dir)
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_samples_ddim)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        
                        # for i in range(grid.size(0)):
                            # save_image(grid[i, :, :, :], os.path.join(outpath,opt.prompt+'_{}.png'.format(i)))
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}-{grid_count:04}.jpg'))
                        grid_count += 1
                    
                    

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    