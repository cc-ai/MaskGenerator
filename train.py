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

    root = Path(__file__).parent.resolve()
    opt_file = "shared/feature_pixelDA.yml"
    opts = load_opts(path=root / opt_file, default=root / "shared/defaults.yml")
    opts = set_mode("train", opts)
    flats = flatten_opts(opts)
    print_opts(flats)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workspace", default=opts.comet.workspace, help="Comet Workspace"
    )
    parser.add_argument(
        "-p",
        "--project_name",
        default=opts.comet.project_name,
        help="Comet project_name",
    )
    parser.add_argument(
        "-n",
        "--no_check",
        action="store_true",
        default=False,
        help="Prevent sample existence checking for faster dev",
    )
    args = parser.parse_args()

    comet_exp = Experiment(workspace=args.workspace, project_name=args.project_name)
    comet_exp.log_asset(file_data=str(root / opt_file), file_name=root / opt_file)
    comet_exp.log_parameters(flats)

    print("Creating loaders:")
    # ! important to do test first
    val_opt = set_mode("test", opts)
    val_loader = get_loader(val_opt, real=True, no_check=args.no_check)
    val_iter = iter(val_loader)
    test_display_images = [
        Dict(val_iter.next()) for i in range(opts.comet.display_size)
    ]

    train_loader = get_loader(opts, real=True, no_check=args.no_check)
    train_iter = iter(train_loader)
    train_display_images = [
        Dict(train_iter.next()) for i in range(opts.comet.display_size)
    ]

    print("Loaders created. Creating network:")

    opts.comet.exp = comet_exp
    model: MaskGenerator = create_model(opts)
    model.setup()

    total_steps = 0
    times = deque([0], maxlen=100)
    model_times = deque([0], maxlen=100)
    batch_size = opts.data.loaders.batch_size
    time_str = "Average time per sample at step {} ({}): {:.3f} (model only: {:.3f})"
    checkpoint_directory, image_directory = prepare_sub_folder(opts.train.output_dir)

    print(">>> Starting training <<<")

    for epoch in range(opts.train.epochs):
        for i, data in enumerate(train_loader):
            times.append(time())
            total_steps += batch_size

            model.set_input(Dict(data))
            model.optimize_parameters()

            model_times.append(time() - times[-1])

            if total_steps // batch_size % 100 == 0:
                avg = avg_duration(times, batch_size)
                print(
                    time_str.format(
                        total_steps, epoch, avg, np.mean(model_times) / batch_size
                    )
                )
                model.comet_exp.log_metric("sample_time", avg, step=total_steps)

            if total_steps % opts.val.save_im_freq == 0:
                model.save_test_images(test_display_images, total_steps)

            if total_steps % opts.train.save_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_steps %d)"
                    % (epoch, total_steps)
                )
                save_suffix = "iter_%d" % total_steps
                model.save_networks(save_suffix)
        # model.update_learning_rate()
