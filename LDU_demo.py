import logging
import numpy as np
"""
Plugin own base_model_class, trainer, validator in example_base_model.py
"""
from libs.example_base_model import base_model_class, trainer, validator


if __name__ == "__main__":
    import os
    import sys
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sys.path.append('libs')
    from ldu import learning_ldu_class as ll
    from ensembles import deep_ensemble_class as de
    logging.getLogger().setLevel(logging.INFO)
    s1 = de(n_ensembles=10, device='cpu')
    s2 = ll(alpha_list=np.arange(1.9, 3.9, 0.2).tolist(), device='cpu', learning_rate=0.0008, epochs=5)

    if not os.path.exists(s1.path_ensemble): # from training base model
        ldu_scores = s2.tune_ldu(lambda: s1.gether_ensembles(base_model_class, trainer, validator))
    else: # from ensemble data
        ldu_scores = s2.tune_ldu(lambda: s1.load_ensemble())
