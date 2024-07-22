import pandas as pd
from tools import *


def main():
    disable_warning()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_root = "Model"
    data_root = "data"
    logs_path_bfnn = os.path.join(data_root, 'model_BFNN.csv')
    logs_path_bftn = os.path.join(data_root, 'model_BFTN.csv')
    nt = 64

    model_bfnn, _ = get_model("BFNN", model_root, "Lest3", nt, device)
    model_bftn, _ = get_model("BFTN", model_root, "Lest3", nt, device)

    print(model_bfnn)
    print(model_bftn)

    model_structure_bfnn = get_model_structure(model_bfnn)
    model_structure_bftn = get_model_structure(model_bftn)

    df_bfnn = pd.DataFrame(model_structure_bfnn, columns=['Layer Name', 'Number of Parameters'])
    df_bftn = pd.DataFrame(model_structure_bftn, columns=['Layer Name', 'Number of Parameters'])

    df_bfnn.to_csv(logs_path_bfnn, index=False)
    df_bftn.to_csv(logs_path_bftn, index=False)


if __name__ == '__main__':
    main()
