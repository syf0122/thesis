from sphericalunet.utils.utils import *

neigh_orders = Get_neighs_order()
print(len(neigh_orders))
for n in neigh_orders:
    print(len(n))
    print(n.max())
    print(n.min())
