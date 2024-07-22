from tools import *


def main():
    args = parse_args()
    model_mode = args.model_mode
    csv_root = "data"
    fig_root = "Figs"

    # read from= csv
    csv_data = read_from_csv(csv_root, model_mode)

    # model/SNR
    plot_data_chinese(csv_data[:6], '训练环境信噪比', fig_root, model_mode)
    # plot_data(csv_data[:6], 'SNR', fig_root, model_mode)

    # model/Lest
    plot_data_chinese(csv_data[6:], '信道路径数量', fig_root, model_mode)
    # plot_data(csv_data[6:], 'Lest', fig_root, model_mode)


if __name__ == '__main__':
    main()
