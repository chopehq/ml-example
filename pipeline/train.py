import sys
import click

from tqdm.autonotebook import tqdm
import pickle
from pathlib import Path

tqdm.pandas()

import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

import logging

logger = logging.getLogger("rec.train")
shandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
shandler.setFormatter(formatter)
logger.addHandler(shandler)
logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--input-reservation-path",
    required=True,
    type=click.Path(),
    help="Path to reservations.csv",
)
@click.option(
    "--output-model-dir",
    required=True,
    type=click.Path(),
    help="Directory to store model artifacts",
)
@click.option("--lfm-num-components", type=int, default=10)
@click.option("--lfm-learning-schedule", type=str, default="adagrad")
@click.option("--lfm-learning-rate", type=float, default=0.01)
@click.option("--lfm-loss", type=str, default="warp")
@click.option("--lfm-epochs", type=int, default=3)
def main(
    input_reservation_path,
    output_model_dir,
    lfm_num_components,
    lfm_learning_schedule,
    lfm_learning_rate,
    lfm_loss,
    lfm_epochs,
):
    logger.info("Loading reservations.csv...")
    rez_df = pd.read_csv(input_reservation_path, dtype={"rez_id": str})

    rez_df["reservation_time"] = pd.to_datetime(
        rez_df["reservation_time"], unit="s", utc=True
    )
    rez_df["booking_time"] = pd.to_datetime(rez_df["booking_time"], unit="s", utc=True)

    logger.info(">> rez_df:")
    print(rez_df)

    logger.info("Calculating rating...")
    rating_df = rez_df.groupby(["hashed_email", "RestaurantUID"]).agg(
        {"rez_id": ["nunique"]}
    )
    rating_df.columns = ["_".join(col).strip() for col in rating_df.columns.values]
    rating_df = rating_df.reset_index()
    rating_df["rating"] = rating_df["rez_id_nunique"].clip(lower=1, upper=5)
    logger.info(">> rating_df:")
    logger.info(rating_df)

    logger.info("Preparing rating format to fit into model...")
    rating_col = "rating"
    cols = ["hashed_email", "RestaurantUID", rating_col]
    rating_dicts = rating_df[cols].to_dict(orient="records")

    logger.info("Constructing LightFM dataset...")
    dataset = Dataset()
    dataset.fit(
        (x["hashed_email"] for x in rating_dicts),
        (x["RestaurantUID"] for x in rating_dicts),
    )

    num_users, num_items = dataset.interactions_shape()
    logger.info(">> Num users: {}, Num restaurants: {}.".format(num_users, num_items))

    (interactions, weights) = dataset.build_interactions(
        ((x["hashed_email"], x["RestaurantUID"], x[rating_col]) for x in rating_dicts)
    )

    logger.info(">> interactions.shape")
    logger.info(repr(interactions))

    logger.info("Fitting LightFM...")
    model = LightFM(
        no_components=lfm_num_components,
        learning_schedule=lfm_learning_schedule,
        learning_rate=lfm_learning_rate,
        loss=lfm_loss,
    )

    model.fit_partial(weights, epochs=lfm_epochs, sample_weight=weights, verbose=True)

    logger.info("Fitting Popular Recommender...")
    pop_res = rez_df.groupby(["RestaurantUID"]).agg({"rez_id": ["nunique"]})
    pop_res.columns = ["_".join(cols) for cols in pop_res.columns]
    pop_rec_df = pop_res.sort_values(["rez_id_nunique"], ascending=False).iloc[:10]
    pop_rec = list(pop_rec_df.index)

    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Persisting LightFM model and dataset and PopRec to {output_model_dir}..."
    )
    with open(output_model_dir + "lfm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_model_dir + "lfm_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open(output_model_dir + "pop_rec.pkl", "wb") as f:
        pickle.dump(pop_rec, f)


if __name__ == "__main__":
    main()
