import torch
import numpy as np
import qpth
import osqp
from scipy import sparse
import pandas as pd
import time
import tqdm
from jrl.math_utils import QP
import argparse
from jrl.config import DEVICE

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default=DEVICE)
args = argparser.parse_args()


def main():
    torch.set_default_device(args.device)

    dimrange = range(2, 50, 5)
    torch.manual_seed(1232)
    ntrials = 5
    df = pd.DataFrame(
        columns=[
            "dim",
            "batch_size",
            "trial_id",
            "oursol_time",
            "oursol_err",
            "qpthsol_time",
            "qpthsol_err",
            "osqp_time",
            "osqp_err",
        ]
    )

    for dim in tqdm.tqdm(dimrange):
        for batch_size in [1, 4, 32, 256]:
            for trial in range(ntrials):
                Q = torch.eye(dim).expand(batch_size, dim, dim)
                p = 2 * torch.ones((batch_size, dim))

                nc = 2 * dim
                G = torch.randn((batch_size, nc, dim))
                h = torch.ones((batch_size, nc))

                start = time.time()
                qp = QP(2 * Q, p, G, h, None, None)
                oursol = qp.solve()
                oursoltime = time.time() - start

                start = time.time()
                qpthsol = qpth.qp.QPFunction(verbose=False)(2 * Q, p, G, h, torch.Tensor(), torch.Tensor())
                qpthsoltime = time.time() - start

                start = time.time()
                osqpsol = torch.zeros_like(oursol)
                for i in range(batch_size):
                    prob = osqp.OSQP()
                    prob.setup(
                        sparse.csc_matrix(2 * Q[i].cpu().numpy()),
                        p[i].cpu().numpy(),
                        sparse.csc_matrix(G[i].cpu().numpy()),
                        -np.inf * np.ones_like(h[i].cpu().numpy()),
                        h[i].cpu().numpy(),
                        verbose=False,
                    )
                    osqpsol[i] = torch.tensor(prob.solve().x.astype(np.float32))
                osqpsoltime = time.time() - start

                osqperror = 0
                # Compute error as mean squared error across batch
                qptherror = torch.mean((qpthsol - osqpsol) ** 2).cpu().numpy()
                ourerror = torch.mean((oursol - osqpsol) ** 2).cpu().numpy()

                df.loc[len(df)] = [
                    dim,
                    batch_size,
                    trial,
                    oursoltime,
                    ourerror,
                    qpthsoltime,
                    qptherror,
                    osqpsoltime,
                    osqperror,
                ]

    df.to_csv("qp_benchmark.csv")


if __name__ == "__main__":
    main()
