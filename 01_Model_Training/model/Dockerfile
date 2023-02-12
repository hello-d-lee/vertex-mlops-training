FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6
WORKDIR /
## Copies the trainer code to the docker image
COPY trainer /trainer
## Sets up the entry point to invoke the trainer
RUN pip3 install -r ./trainer/requirements.txt
ENTRYPOINT ["python", "-m", "trainer.model"]