import time
from pathlib import Path
import sys
from utils import *
from data.datasets import get_loader

# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()

    opt = load_opts(path=root / "example_files/testing.yml", default=root / "shared/defaults.yml")
    opt = set_mode("train", opt)
    loader = get_loader(opt)
    dataset_size = len(loader)
    print("#training images = %d" % dataset_size)

    model = create_model(opt)
    model.setup()

    total_steps = 0

    for epoch in range(opt.train.epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(loader):
            iter_start_time = time.time()
            if total_steps % opt.train.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.data.loaders.batch_size
            epoch_iter += opt.data.loaders.batch_size

            model.set_input(Dict(data))
            model.optimize_parameters()
            """
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, opt, losses
                    )

            if total_steps % opt.save_latest_freq == 0:
                print("saving the latest model (epoch %d, total_steps %d)" % (epoch, total_steps))
                save_suffix = "iter_%d" % total_steps if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
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

