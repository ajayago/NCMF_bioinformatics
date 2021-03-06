import sys
import getopt
from utils import *
from grgmf import GRGMF
import pandas as pd
import logging
logging.basicConfig(level='WARNING')


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "d:o", ["dataset=", "method-options=", ])
    except getopt.GetoptError:
        sys.exit()
    data_dir = './data/'
    output_dir = './output'
    model_settings = []

    for opt, arg in opts:
        if opt == "--dataset":
            dataset = arg
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    args = {'k': 5, 'max_iter': 100, 'lr': 0.1, 'lamb': 0.0333, 'beta': 4., 'r1': 0.5, 'r2': 1.,
            'K': 150, 'c': 5, 'eta': 0.5}
    for key, val in model_settings:
        args[key] = eval(val)

    intMat, A_sim, B_sim = load_data_from_file(dataset, data_dir)
    A_names, B_names = get_names(dataset, data_dir)

    model = GRGMF(max_iter=args['max_iter'], c=args['c'], lamb=args['lamb'], beta=args['beta'],
                  r1=args['r1'], r2=args['r2'], lr=args['lr'], K=args['K'], k=args['k'], eta=args['eta'])
    cmd = str(model)
    logging.info(("Dataset:" + dataset + "\n" + cmd))
    W = np.ones(intMat.shape)
    W[:, np.where(intMat.sum(0) == 0)] = 0
    W[np.where(intMat.sum(1) == 0), :] = 0
    model.fix_model(W, intMat, A_sim, B_sim, 22)
    x, y = np.where(intMat == 0)
    scores = model.predict_scores(list(zip(x, y)), 5)
    ii = np.argsort(scores)[::-1]
    assert (len(A_names), len(B_names)) == intMat.shape
    predict_pairs = [(A_names[x[i]], B_names[y[i]], scores[i]) for i in ii]
    novel_pairs = pd.DataFrame(predict_pairs)
    novel_pairs.columns = ['A', 'B', 'Probability']
    predicted_matrix = pd.DataFrame(data=model.P.copy(), index=A_names, columns=B_names)

    novel_pairs.to_excel(os.path.join(output_dir, dataset + '_novel.xlsx'))
    predicted_matrix.to_excel(os.path.join(output_dir, dataset + '_predicted_matrix.xlsx'))

    U, V = model.get_repr()
    print(U.size())
    print(V.size())
    np.save("PolyP3_2_drug_repr.npy", U.cpu().detach().numpy())
    np.save("PolyP3_2_protein_repr.npy", V.cpu().detach().numpy())
