FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110

USER root:root

RUN pip install pybind11 regex

RUN pip install git+https://github.com/microsoft/DeepSpeed.git
