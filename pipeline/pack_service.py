import sys
import pickle
import click

from bento_lfm_service import LightFMRecService

import logging

logger = logging.getLogger("rec.serve")
shandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
shandler.setFormatter(formatter)
logger.addHandler(shandler)
logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--output-model-dir",
    required=True,
    type=click.Path(),
    help="Directory to store model artifacts",
)
@click.option("--service-version", required=True, type=str)
def main(output_model_dir, service_version):

    logger.info(
        f"Loading LightFM model and dataset and PopRec from {output_model_dir}..."
    )
    with open(output_model_dir + "lfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(output_model_dir + "lfm_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(output_model_dir + "pop_rec.pkl", "rb") as f:
        pop_rec = pickle.load(f)

    lfm_service = LightFMRecService()

    lfm_service.pack("model", model)
    lfm_service.pack("dataset", dataset)
    lfm_service.pack("pop_rec", pop_rec)

    lfm_service.save(version=service_version)


if __name__ == "__main__":
    main()
