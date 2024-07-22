# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/17 0:22
# @Function:
from tools import *
from torch import optim
from Enumber import Environment
from torch.utils.tensorboard import SummaryWriter


def main():
    args = parse_args()
    model_mode = args.model_mode
    num_epochs = args.epochs
    lr = args.lr

    disable_warning()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_root = "./Model"
    logs_root = "Logs"
    dataset_root = "train_set"

    batch_size = 64
    gamma = 0.1

    nt = 64

    writer = SummaryWriter(os.path.join(logs_root, model_mode))
    output_list = []
    output_list.append(f"model_mode = {model_mode},num_epochs = {num_epochs}, lr = {lr}. {get_time()}")

    # loss function
    criterion = get_loss_function(nt, device)

    for env in Environment:
        model_name = f"{model_mode}_{env.name}.pth"
        model_path = os.path.join(model_root, model_mode, model_name)
        output_list.append(f"{model_name} start to train. {get_time()}")

        # Reading the data
        output_list.append(f"Reading the data. {get_time()}")
        dataset_path = dataset_path_append(dataset_root, env.name)
        train_dataloader, test_dataloader = get_data(dataset_path, batch_size)

        # Create/Read the model
        model, logs_out = get_model(model_mode, model_root, env.name, nt, device)
        output_list.extend(logs_out)

        # Test the original model
        capacity = test(model, test_dataloader, criterion, env, device)
        best_model, best_capacity = update_best_model(model, capacity)
        output_list.append(f"Original Model's capacity : {capacity}. {get_time()}")

        # optimizer
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        # scheduler
        milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

        # train && test
        for epoch in range(num_epochs):
            start_time = time.time()

            _ = train(model, train_dataloader, criterion, optimizer, scheduler, env, device)
            capacity = test(model, test_dataloader, criterion, env, device)

            end_time = time.time()
            epoch_time = end_time - start_time

            # best model update
            if capacity > best_capacity:
                best_model, best_capacity = update_best_model(model, capacity)
                output_list.append(f"New best model marked, best_capacity={best_capacity}, epoch= {epoch+ 1}. {get_time()}")

            writer.add_scalar(f"{model_mode}_{env.name}'s capacity", capacity, epoch)

            output_list.append(
                'Epoch [{}/{}],Test Capacity: {:.4f}, Epoch Time: {:.4f}s'.format(epoch + 1, num_epochs, capacity,
                                                                                  epoch_time))
        torch.save(best_model, model_path)

    writer.close()
    write2json(logs_root, output_list)


if __name__ == '__main__':
    main()
