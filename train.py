from comet_ml import Experiment
from time import time
from pathlib import Path
from addict import Dict
import numpy as np
from utils import (
    load_opts,
    set_mode,
    prepare_sub_folder,
    create_model,
    avg_duration,
    flatten_opts,
    print_opts,
)
from data.datasets import get_loader
from collections import deque
import argparse
from models.mask_generator import MaskGenerator


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse Arguments  -----
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace", help="Comet Workspace")
    parser.add_argument("-p", "--project_name", help="Comet project_name")
    parser.add_argument(
        "-n",
        "--no_check",
        action="store_true",
        default=False,
        help="Prevent sample existence checking for faster dev",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Config file to use",
        default="shared/feature_pixelDA.yml",
    )
    args = Dict(vars(parser.parse_args()))

    # --------------------------
    # -----  Load Options  -----
    # --------------------------
    root = Path(__file__).parent.resolve()
    opts = load_opts(path=root / args.config, default=root / "shared/defaults.yml")
    opts = set_mode("train", opts)
    flats = flatten_opts(opts)
    print_opts(flats)

    # ------------------------------------
    # -----  Start Comet Experiment  -----
    # ------------------------------------
    wsp = args.get("workspace") or opts.comet.workspace
    prn = args.get("project_name") or opts.comet.project_name
    comet_exp = Experiment(workspace=wsp, project_name=prn)
    comet_exp.log_asset(file_data=str(root / args.config), file_name=root / args.config)
    comet_exp.log_parameters(flats)

    # ----------------------------
    # -----  Create loaders  -----
    # ----------------------------
    print("Creating loaders:")
    # ! important to do test first
    val_opt = set_mode("test", opts)
    val_loader = get_loader(val_opt, real=True, no_check=args.no_check)
    train_loader = get_loader(opts, real=True, no_check=args.no_check)
    print("Creating display images...", end="", flush=True)
    val_iter = iter(val_loader)

    test_display_images = [
        Dict(val_loader.dataset[i]) for i in range(opts.comet.display_size)
    ]
    print(Dict(val_loader.dataset[0]).data.x.shape)
    if opts.train.save_im:
        train_display_images = [
            Dict(train_loader.dataset[i]) for i in range(opts.comet.display_size)
        ]

    print("ok.")

    # --------------------------
    # -----  Create Model  -----
    # --------------------------
    print("Creating Model:")
    opts.comet.exp = comet_exp
    model: MaskGenerator = create_model(opts)
    model.setup()

    # ---------------------------
    # -----  Miscellaneous  -----
    # ---------------------------
    total_steps = 0
    times = deque([0], maxlen=100)
    model_times = deque([0], maxlen=100)
    batch_size = opts.data.loaders.batch_size
    checkpoint_directory, image_directory = prepare_sub_folder(opts.train.output_dir)
    tpe = opts.train.tests_per_epoch
    test_idx = [i * len(train_loader) // tpe for i in range(tpe)]
    test_idx[-1] = len(train_loader) - 1
    test_idx = set(test_idx)

    # ---------------------------
    # -----  Training Loop  -----
    # ---------------------------
    s = "Starting training for {} epochs of {} updates with batch size {}, "
    s += "{} test inferences per epoch."
    print(s.format(opts.train.epochs, len(train_loader), batch_size, tpe))

    for epoch in range(opts.train.epochs):
        print(f"Epoch {epoch}: ")
        comet_exp.log_metric("epoch", epoch, step=total_steps)
        for i, data in enumerate(train_loader):
            times.append(time())
            total_steps += batch_size

            model.set_input(Dict(data))
            model.optimize_parameters(total_steps)

            model_times.append(time() - times[-1])
            if total_steps // batch_size % 100 == 0:
                avg = avg_duration(times, batch_size)
                mod_times = np.mean(model_times) / batch_size
                comet_exp.log_metric("sample_time", avg, step=total_steps)
                comet_exp.log_metric("model_time", mod_times, step=total_steps)
            if i in test_idx or total_steps == batch_size:
                print(f"({total_steps}) Inferring test images...", end="", flush=True)
                t = model.save_test_images(test_display_images, total_steps)
                print("ok in {:.2f}s.".format(t))

                if opts.train.save_im:
                    print(
                        f"({total_steps}) Inferring train images...", end="", flush=True
                    )
                    t = model.save_test_images(
                        train_display_images, total_steps, is_test=False
                    )
                    print("ok in {:.2f}s.".format(t))

        print("saving (epoch %d, total_steps %d)" % (epoch, total_steps))
        save_suffix = "iter_%d" % total_steps
        model.save_networks(save_suffix)
        # model.update_learning_rate()
