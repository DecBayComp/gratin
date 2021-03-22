FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.7

RUN mkdir Docker/

RUN apt-get -y update
RUN apt-get -y install openssh-client

COPY docker/requirements.txt Docker/requirements.txt
COPY docker/torch_geometric.txt Docker/torch_geometric.txt
COPY docker/install_jupyter_extensions.txt Docker/install_jupyter_extensions.txt
COPY docker/install_jax.sh Docker/install_jax.sh

RUN pip3 install -r Docker/requirements.txt
RUN pip3 install --no-index \
    -r Docker/torch_geometric.txt \
    -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
RUN pip3 install torch-geometric

RUN chmod +x Docker/install_jupyter_extensions.txt
RUN ./Docker/install_jupyter_extensions.txt

RUN chmod +x Docker/install_jax.sh
RUN Docker/install_jax.sh

RUN rm -r Docker

WORKDIR /gratin

#COPY . .

#CMD ["jupyter","notebook","--allow-root","--no-browser","--port=8800","--ip=0.0.0.0"]
