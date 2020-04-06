import time
from pathlib import Path
from comet_ml import Experiment
import sys
from utils import *
from data.datasets import get_loader
import copy

# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config",
    type=str,
    default="shared/sim.yaml",
    help="Path to the config file.",
    )  
    opts = parser.parse_args()

    root = Path(__file__).parent.resolve()
    opt_file = opts.config

    opt = load_opts(path=root / opt_file, default=root / "shared/defaults.yml")

    #! important to do test first
    val_opt = set_mode("test", opt)
    val_loader = get_loader(val_opt)
    test_display_images = [Dict(iter(val_loader).next()) for i in range(opt.comet.display_size)]

    opt = set_mode("train", opt)
    loader = get_loader(opt)
    train_display_images = [Dict(iter(loader).next()) for i in range(opt.comet.display_size)]

    dataset_size = len(loader)
    print("#training images = %d" % dataset_size)

    comet_exp = Experiment(workspace=opt.comet.workspace, project_name=opt.comet.project_name)
    if comet_exp is not None:
        comet_exp.log_asset(file_data=str(root / opt_file), file_name=root / opt_file)
        comet_exp.log_parameters(opt)

    checkpoint_directory, image_directory = prepare_sub_folder(opt.train.output_dir)

    opt.comet.exp = comet_exp

    model = create_model(opt)
    model.setup()

    total_steps = 0

    for epoch in range(opt.train.epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(loader):
            with Timer("Elapsed time in update " + str(i) + ": %f"):
                iter_start_time = time.time()
                if total_steps % opt.train.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.data.loaders.batch_size
                epoch_iter += opt.data.loaders.batch_size

                model.set_input(Dict(data))
                model.optimize_parameters()

                if total_steps % opt.val.save_im_freq == 0:
                    model.save_test_images(test_display_images, total_steps, test = True)
                if total_steps % opt.train.save_im_freq == 0:    
                    model.save_test_images(train_display_images, total_steps, test = False)

                if total_steps % opt.train.save_freq == 0:
                    print(
                        "saving the latest model (epoch %d, total_steps %d)" % (epoch, total_steps)
                    )
                    save_suffix = "iter_%d" % total_steps
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

        """
        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d, iters %d" % (epoch, total_steps))
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()
        """

