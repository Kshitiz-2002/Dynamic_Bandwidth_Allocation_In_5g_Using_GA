## Example Usage

Access-Aware

```python
from Algorithms.Access_Aware import AccessAware
from typing import List

numMBS: int = 3
numMC: int = 10
nMC: List[int] = [4, 3, 3]
B: List[int] = [0, 0, 0]
R: List[List[float]] = [[10.0, 20.0, 15.0, 5.0], [25.0, 30.0, 15.0], [20.0, 10.0, 5.0]]
bm: float = 5.0  # Minimum frequency resource
bM: float = 100.0  # Maximum BW

aa: AccessAware = AccessAware(numMBS, numMC, nMC, B, R, bm, bM)
S: float = 1.0  # Example signal power
N: float = 0.1  # Example noise power
Bi, bij = aa.dynamic_allocation(S, N)

print("Total assigned bandwidth to MBSs:", Bi)
print("Bandwidth assigned to each cell:", bij)
```