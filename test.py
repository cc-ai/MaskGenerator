from pathlib import Path

from addict import Dict
from comet_ml import Experiment

from data.datasets import get_loader
from utils import Timer, create_model, load_opts, prepare_sub_folder, set_mode

# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    opt_file = "prev_experiments/11k_wgan_feature_pixelDA.yml"

    opts = load_opts(path=root / opt_file, default=root / "shared/defaults.yml")

    opts = set_mode("test", opts)
    opts.data.loaders.batch_size = 1
    val_loader = get_loader(opts)
    dataset_size = len(val_loader)

    print("#testing images = %d" % dataset_size)

    comet_exp = Experiment(
        workspace=opts.comet.workspace, project_name=opts.comet.project_name
    )
    if comet_exp is not None:
        comet_exp.log_asset(file_data=str(root / opt_file), file_name=root / opt_file)
        comet_exp.log_parameters(opts)

    checkpoint_directory, image_directory = prepare_sub_folder(opts.train.output_dir)

    opts.comet.exp = comet_exp

    model = create_model(opts)
    model.setup()

    total_steps = 0

    for i, data in enumerate(val_loader):
        with Timer("Elapsed time in update " + str(i) + ": %f"):
            total_steps += opts.data.loaders.batch_size
            model.set_input(Dict(data))
            model.save_test_images([Dict(data)], total_steps)
