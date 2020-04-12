# For more information on creating a Dockerfile
# https://github.com/awslabs/amazon-sagemaker-examples/
ARG REGION=us-west-1
ARG CONTEXT=gpu-py36-cu101
ARG DLAMI_REGISTRY_ID=763104351884

# SageMaker PyTorch image
# Registry ID from: https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html
FROM ${DLAMI_REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:1.4.0-${CONTEXT}-ubuntu16.04

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# Need to pre-install `cython` for ConfigSpace.
RUN python -m pip install --upgrade pip
RUN python -m pip install Cython==0.29.13

RUN git clone https://github.com/StacyYang/AutoTorch autotorch-docker
RUN python -m pip install -e autotorch-docker/

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /examples /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM sagemaker_example.py
