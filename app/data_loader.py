
import csv
from pathlib import Path

# Resolve data folder relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_csv(filename):
    file_path = DATA_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


customers = load_csv("customers.csv")
transactions = load_csv("transactions.csv")


def find_customer(customer_id):
    return next(
        (
            customer for customer in customers
            if customer["customer_id"] == customer_id
        ),
        None
    )


def find_transactions(customer_id):
    return [
        transaction for transaction in transactions
        if transaction["customer_id"] == customer_id
    ]
