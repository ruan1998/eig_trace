from utils.get_eig_values import Eigenval
import pandas as pd
# import time
# for i in range(20):
#     start_time = time.time()
#     print(Eigenval(i))
#     print('[+]:',time.time()-start_time)

to_run_df = pd.DataFrame(index=range(96*87), columns=['real','imag','time'])
result = Eigenval(0)
to_run_df.loc[:86,'real'] = result.real
to_run_df.loc[:86,'imag'] = result.imag
to_run_df.loc[:86,'time'] = [0]*87
print(to_run_df.loc[:86,:])
