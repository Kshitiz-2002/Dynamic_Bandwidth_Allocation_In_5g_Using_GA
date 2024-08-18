import math
from typing import List, Tuple


def capacity(bij: float, S: float, N: float) -> float:
    return bij * math.log2(1 + S / (N + 1))


class AccessAware:
    def __init__(self, numMBS: int, numMC: int, nMC: List[int], B: List[int], R: List[List[float]], bm: float, bM: float):
        self.numMBS: int = numMBS
        self.numMC: int = numMC
        self.nMC: List[int] = nMC
        self.B: List[int] = B
        self.R: List[List[float]] = R
        self.bm: float = bm
        self.bM: float = bM

    def dynamic_allocation(self, S: float, N: float) -> Tuple[List[float], List[List[float]]]:
        Bi: List[float] = [0 for _ in range(self.numMBS)]
        bij: List[List[float]] = [[0 for _ in range(self.nMC[i])] for i in range(self.numMBS)]
        allocated: int = 0

        while allocated < sum(self.nMC):
            for i in range(self.numMBS):
                sorted_MC: List[Tuple[int, float]] = sorted(enumerate(self.R[i]), key=lambda x: x[1], reverse=True)
                satisfied: int = 0

                for j, rij in sorted_MC[:self.nMC[i]]:
                    if Bi[i] + self.bm <= self.bM and rij > capacity(bij=bij[i][j], S=S, N=N):
                        bij[i][j] += self.bm
                        Bi[i] += self.bm
                        cij: float = capacity(bij=bij[i][j], S=S, N=N)

                        if cij >= rij:
                            allocated += 1
                            satisfied += 1

                if Bi[i] == self.bM:
                    allocated += self.nMC[i] - satisfied

        return Bi, bij
