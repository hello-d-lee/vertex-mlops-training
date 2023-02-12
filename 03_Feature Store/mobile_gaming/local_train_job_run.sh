
gcloud ai custom-jobs local-run --executor-image-uri=us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest --python-module=trainer.task --extra-dirs=config,data,model -- --data_path data/raw/sample.csv --model_dir model --config_path config/config.yaml
