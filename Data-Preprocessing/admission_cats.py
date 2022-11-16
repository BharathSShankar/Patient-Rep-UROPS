import pandas as pd
import numpy as np

adm_dat = pd.read_csv("../data/patientData/ADMISSIONS.csv")

np.save("../data/dataByPatient/eth_cats.npy", adm_dat["ETHNICITY"].unique())
np.save("../data/dataByPatient/lang_cats.npy", adm_dat["LANGUAGE"].unique())
np.save("../data/dataByPatient/mar_cats.npy", adm_dat["MARITAL_STATUS"].unique())
np.save("../data/dataByPatient/ins_cats.npy", adm_dat["INSURANCE"].unique())
np.save("../data/dataByPatient/rel_cats.npy", adm_dat["RELIGION"].unique())
