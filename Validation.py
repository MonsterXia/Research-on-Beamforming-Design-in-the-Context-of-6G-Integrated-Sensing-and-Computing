from tools import *
from Enumber import Environment


def main():
    args = parse_args()
    model_mode = args.model_mode

    disable_warning()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_root = "train_set"
    model_root = "Model"
    csv_root = "data"

    batch_size = 64

    nt = 64

    # loss function
    criterion = get_loss_function(nt, device)

    for env in Environment:
        # Reading the data
        print("Reading the data...")
        dataset_path = dataset_path_append(dataset_root, env.name)
        _, test_dataloader = get_data(dataset_path, batch_size)

        # Create/Read the model
        model, _ = get_model(model_mode, model_root, env.name, nt, device)

        # test
        capacity = vali(model, test_dataloader, criterion, device)

        # write to .csv
        write_to_csv(csv_root, model_mode, env.name, capacity)


if __name__ == '__main__':
    main()
