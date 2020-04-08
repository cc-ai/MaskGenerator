from comet_ml import Experiment
import time
from pathlib import Path
from addict import Dict
from utils import load_opts, set_mode, prepare_sub_folder, create_model, avg_duration
from data.datasets import get_loader
from collections import deque

# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    opt_file = "shared/feature_pixelDA.yml"

    opts = load_opts(path=root / opt_file, default=root / "shared/defaults.yml")
    comet_exp = Experiment(
        workspace=opts.comet.workspace, project_name=opts.comet.project_name
    )

    # ! important to do test first
    val_opt = set_mode("test", opts)
    val_loader = get_loader(val_opt, real=True)
    test_display_images = [
        Dict(iter(val_loader).next()) for i in range(opts.comet.display_size)
    ]

    opts = set_mode("train", opts)
    loader = get_loader(opts, real=True)
    train_display_images = [
        Dict(iter(loader).next()) for i in range(opts.comet.display_size)
    ]

    dataset_size = len(loader)
    print("#training images = %d" % dataset_size)

    if comet_exp is not None:
        comet_exp.log_asset(file_data=str(root / opt_file), file_name=root / opt_file)
        comet_exp.log_parameters(opts)

    checkpoint_directory, image_directory = prepare_sub_folder(opts.train.output_dir)

    opts.comet.exp = comet_exp

    model = create_model(opts)
    model.setup()

    total_steps = 0
    times = deque([0], maxlen=100)
    batch_size = opts.data.loaders.batch_size
    time_str = "Average time per sample at step {}: {:.3f}"

    for epoch in range(opts.train.epochs):
        for i, data in enumerate(loader):
            times.append(time())
            total_steps += batch_size

            model.set_input(Dict(data))
            model.optimize_parameters()

            if total_steps // batch_size % 25 == 0:
                avg = avg_duration(times)
                print(time_str.format(total_steps, avg))
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
