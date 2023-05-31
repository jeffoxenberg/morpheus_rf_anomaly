FROM jupyter/datascience-notebook:hub-4.0.0
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#add-a-custom-conda-environment-and-jupyter-kernel
RUN conda config --add channels nvidia
RUN conda config --add channels rapidsai
RUN mamba install -y cudatoolkit=11.4 cudf cuml cupy && fix-permissions "${CONDA_DIR}" && fix-permissions "/home/${NB_USER}"
RUN pip install tritonclient[all] xgboost