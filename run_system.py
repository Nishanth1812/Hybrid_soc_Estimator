from validation.dataset_validator import DatasetValidator
from training.train_pipeline import train_model


def main():

    dataset_path = "datasets/processed"

    validator = DatasetValidator(dataset_path)

    validator.run_full_validation()

    print("Starting training...")

    model = train_model(dataset_path)

    print("Training finished")


if __name__ == "__main__":

    main()