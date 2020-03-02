FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get install -y graphviz

RUN pip install --upgrade pip
RUN pip install tensorflow_datasets \
                tensorflow-addons \
                ipython \
                jupyterlab \
                matplotlib \
                pydot \
                graphviz \
                comet_ml

CMD ["python"]
