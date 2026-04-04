#!/usr/bin/env python3
"""
Generate diverse messy+clean CSV pairs for training.

Creates paired files across multiple domains with controlled data quality issues.
Each domain has a clean version and a messy version with planted issues.

Usage:
    python generate_training_data.py --output-dir data_clean_env/tasks --num-samples 50
"""

import argparse
import csv
import os
import random
import string
from datetime import datetime, timedelta


def random_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def corrupt_date(date_str):
    """Corrupt a date string in various ways."""
    if not date_str:
        return ""
    fmt = random.choice(["slash", "dash_mixed", "us", "short", "euro"])
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if fmt == "slash":
            return dt.strftime("%m/%d/%Y")
        elif fmt == "dash_mixed":
            return dt.strftime("%m-%d-%Y")
        elif fmt == "us":
            return dt.strftime("%-m/%-d/%y")
        elif fmt == "short":
            return dt.strftime("%-m/%-d/%y")
        elif fmt == "euro":
            return dt.strftime("%d-%m-%Y")
    except:
        return date_str
    return date_str


def corrupt_name(name):
    """Corrupt a name with casing and whitespace issues."""
    if not name:
        return ""
    fmt = random.choice(["upper", "lower", "extra_space", "no_change"])
    if fmt == "upper":
        return name.upper()
    elif fmt == "lower":
        return name.lower()
    elif fmt == "extra_space":
        return f"  {name}  "
    return name


def corrupt_email(email):
    """Corrupt an email in various ways."""
    if not email:
        return ""
    fmt = random.choice(
        ["upper", "missing_at", "missing_domain", "extra_space", "no_change"]
    )
    if fmt == "upper":
        return email.upper()
    elif fmt == "missing_at":
        return email.replace("@", "")
    elif fmt == "missing_domain":
        return email.split("@")[0] + "@example"
    elif fmt == "extra_space":
        return f"  {email}  "
    return email


def corrupt_price(price_str):
    """Corrupt a price string."""
    if not price_str:
        return ""
    fmt = random.choice(["dollar", "usd", "extra_space", "no_change"])
    if fmt == "dollar":
        return f"${price_str}"
    elif fmt == "usd":
        return f"{price_str} USD"
    elif fmt == "extra_space":
        return f"  ${price_str}  "
    return price_str


def maybe_missing(prob=0.1):
    """Randomly return empty string with given probability."""
    if random.random() < prob:
        return ""
    return None


def generate_hr_data(n_rows=30, seed=None):
    """Generate HR employee records."""
    if seed:
        random.seed(seed)

    first_names = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "Jack",
        "Karen",
        "Leo",
        "Mia",
        "Noah",
        "Olivia",
        "Paul",
        "Quinn",
        "Rosa",
        "Sam",
        "Tina",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Wilson",
        "Moore",
        "Taylor",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Martin",
        "Thompson",
        "Robinson",
        "Clark",
    ]
    departments = [
        "Engineering",
        "Marketing",
        "Sales",
        "HR",
        "Finance",
        "Operations",
        "Legal",
        "IT",
    ]
    cities = [
        "New York",
        "London",
        "Tokyo",
        "Berlin",
        "Sydney",
        "Toronto",
        "Paris",
        "Mumbai",
    ]

    clean_rows = []
    for i in range(n_rows):
        row = {
            "employee_id": f"EMP{i + 1:04d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "department": random.choice(departments),
            "salary": f"{random.randint(40000, 150000)}",
            "hire_date": random_date().strftime("%Y-%m-%d"),
            "city": random.choice(cities),
            "email": f"{random.choice(first_names).lower()}.{random.choice(last_names).lower()}@company.com",
        }
        clean_rows.append(row)

    messy_rows = []
    for row in clean_rows:
        messy = dict(row)
        # Add duplicates (5% chance)
        if random.random() < 0.05:
            messy_rows.append(dict(row))  # Add duplicate before corruption

        # Corrupt fields
        messy["name"] = corrupt_name(messy["name"])
        messy["department"] = random.choice(
            [
                messy["department"].upper(),
                messy["department"].lower(),
                f"  {messy['department']}  ",
                messy["department"],
            ]
        )
        messy["salary"] = corrupt_price(messy["salary"])
        messy["hire_date"] = corrupt_date(messy["hire_date"])
        messy["email"] = corrupt_email(messy["email"])

        # Add missing values
        for field in ["salary", "email", "city"]:
            if maybe_missing(0.08) is not None:
                messy[field] = ""

        messy_rows.append(messy)

    return clean_rows, messy_rows


def generate_healthcare_data(n_rows=30, seed=None):
    """Generate patient appointment records."""
    if seed:
        random.seed(seed)

    first_names = [
        "James",
        "Mary",
        "Robert",
        "Patricia",
        "John",
        "Jennifer",
        "Michael",
        "Linda",
        "David",
        "Elizabeth",
        "William",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
    ]
    last_names = [
        "Martinez",
        "Robinson",
        "Lewis",
        "Lee",
        "Walker",
        "Hall",
        "Allen",
        "Young",
        "King",
        "Wright",
    ]
    conditions = [
        "Diabetes",
        "Hypertension",
        "Asthma",
        "Arthritis",
        "Migraine",
        "Anxiety",
        "Depression",
        "Back Pain",
    ]
    doctors = [
        "Dr. Smith",
        "Dr. Patel",
        "Dr. Kim",
        "Dr. Garcia",
        "Dr. Chen",
        "Dr. Brown",
        "Dr. Lee",
        "Dr. Davis",
    ]

    clean_rows = []
    for i in range(n_rows):
        row = {
            "patient_id": f"PT{i + 1:05d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "condition": random.choice(conditions),
            "appointment_date": random_date().strftime("%Y-%m-%d"),
            "doctor": random.choice(doctors),
            "insurance_id": f"INS-{random.randint(100000, 999999)}",
            "status": random.choice(["scheduled", "completed", "cancelled", "no-show"]),
        }
        clean_rows.append(row)

    messy_rows = []
    for row in clean_rows:
        messy = dict(row)
        if random.random() < 0.05:
            messy_rows.append(dict(row))

        messy["name"] = corrupt_name(messy["name"])
        messy["condition"] = random.choice(
            [
                messy["condition"].upper(),
                messy["condition"].lower(),
                f"  {messy['condition']}  ",
                messy["condition"],
            ]
        )
        messy["appointment_date"] = corrupt_date(messy["appointment_date"])
        messy["doctor"] = random.choice(
            [
                messy["doctor"].upper(),
                messy["doctor"].lower(),
                f"  {messy['doctor']}  ",
                messy["doctor"],
            ]
        )
        messy["status"] = random.choice(
            [
                messy["status"].upper(),
                messy["status"].lower(),
                f"  {messy['status']}  ",
                messy["status"],
            ]
        )

        for field in ["insurance_id", "status"]:
            if maybe_missing(0.08) is not None:
                messy[field] = ""

        messy_rows.append(messy)

    return clean_rows, messy_rows


def generate_finance_data(n_rows=30, seed=None):
    """Generate financial transaction records."""
    if seed:
        random.seed(seed)

    merchants = [
        "Amazon",
        "Walmart",
        "Target",
        "Starbucks",
        "Uber",
        "Netflix",
        "Spotify",
        "Apple",
        "Google",
        "Microsoft",
    ]
    categories = [
        "Shopping",
        "Food",
        "Transport",
        "Entertainment",
        "Utilities",
        "Healthcare",
        "Travel",
        "Education",
    ]
    payment_methods = [
        "Credit Card",
        "Debit Card",
        "Cash",
        "Bank Transfer",
        "Digital Wallet",
    ]

    clean_rows = []
    for i in range(n_rows):
        amount = round(random.uniform(5.0, 500.0), 2)
        row = {
            "transaction_id": f"TXN{i + 1:06d}",
            "date": random_date().strftime("%Y-%m-%d"),
            "merchant": random.choice(merchants),
            "category": random.choice(categories),
            "amount": f"{amount:.2f}",
            "payment_method": random.choice(payment_methods),
            "status": random.choice(["completed", "pending", "refunded", "failed"]),
        }
        clean_rows.append(row)

    messy_rows = []
    for row in clean_rows:
        messy = dict(row)
        if random.random() < 0.05:
            messy_rows.append(dict(row))

        messy["merchant"] = random.choice(
            [
                messy["merchant"].upper(),
                messy["merchant"].lower(),
                f"  {messy['merchant']}  ",
                messy["merchant"],
            ]
        )
        messy["category"] = random.choice(
            [
                messy["category"].upper(),
                messy["category"].lower(),
                f"  {messy['category']}  ",
                messy["category"],
            ]
        )
        messy["amount"] = corrupt_price(messy["amount"])
        messy["date"] = corrupt_date(messy["date"])
        messy["payment_method"] = random.choice(
            [
                messy["payment_method"].upper(),
                messy["payment_method"].lower(),
                f"  {messy['payment_method']}  ",
                messy["payment_method"],
            ]
        )
        messy["status"] = random.choice(
            [
                messy["status"].upper(),
                messy["status"].lower(),
                f"  {messy['status']}  ",
                messy["status"],
            ]
        )

        for field in ["payment_method", "status"]:
            if maybe_missing(0.08) is not None:
                messy[field] = ""

        messy_rows.append(messy)

    return clean_rows, messy_rows


def generate_logistics_data(n_rows=30, seed=None):
    """Generate shipping/delivery records."""
    if seed:
        random.seed(seed)

    origins = [
        "Shanghai",
        "Shenzhen",
        "Mumbai",
        "Hamburg",
        "Los Angeles",
        "Singapore",
        "Rotterdam",
        "Dubai",
    ]
    destinations = [
        "New York",
        "London",
        "Tokyo",
        "Sydney",
        "Dubai",
        "Singapore",
        "Hamburg",
        "Mumbai",
    ]
    carriers = ["FedEx", "DHL", "UPS", "Maersk", "COSCO", "MSC", "CMA CGM", "Evergreen"]
    statuses = ["in_transit", "delivered", "customs", "delayed", "returned"]

    clean_rows = []
    for i in range(n_rows):
        weight = round(random.uniform(0.5, 500.0), 1)
        row = {
            "tracking_id": f"TRK{i + 1:07d}",
            "origin": random.choice(origins),
            "destination": random.choice(destinations),
            "carrier": random.choice(carriers),
            "weight_kg": f"{weight:.1f}",
            "ship_date": random_date().strftime("%Y-%m-%d"),
            "status": random.choice(statuses),
        }
        clean_rows.append(row)

    messy_rows = []
    for row in clean_rows:
        messy = dict(row)
        if random.random() < 0.05:
            messy_rows.append(dict(row))

        messy["origin"] = random.choice(
            [
                messy["origin"].upper(),
                messy["origin"].lower(),
                f"  {messy['origin']}  ",
                messy["origin"],
            ]
        )
        messy["destination"] = random.choice(
            [
                messy["destination"].upper(),
                messy["destination"].lower(),
                f"  {messy['destination']}  ",
                messy["destination"],
            ]
        )
        messy["carrier"] = random.choice(
            [
                messy["carrier"].upper(),
                messy["carrier"].lower(),
                f"  {messy['carrier']}  ",
                messy["carrier"],
            ]
        )
        messy["ship_date"] = corrupt_date(messy["ship_date"])
        messy["status"] = random.choice(
            [
                messy["status"].upper(),
                messy["status"].lower(),
                f"  {messy['status']}  ",
                messy["status"],
            ]
        )

        for field in ["weight_kg", "status"]:
            if maybe_missing(0.08) is not None:
                messy[field] = ""

        messy_rows.append(messy)

    return clean_rows, messy_rows


def generate_education_data(n_rows=30, seed=None):
    """Generate student grade records."""
    if seed:
        random.seed(seed)

    first_names = [
        "Emma",
        "Liam",
        "Sophia",
        "Noah",
        "Olivia",
        "Ava",
        "Lucas",
        "Mia",
        "Ethan",
        "Isabella",
        "Mason",
        "Charlotte",
        "Logan",
        "Amelia",
        "Aiden",
        "Harper",
    ]
    last_names = [
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
    ]
    subjects = [
        "Math",
        "Physics",
        "Chemistry",
        "Biology",
        "English",
        "History",
        "CS",
        "Art",
    ]
    grades = ["A", "B", "C", "D", "F"]

    clean_rows = []
    for i in range(n_rows):
        row = {
            "student_id": f"STU{i + 1:04d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "subject": random.choice(subjects),
            "grade": random.choice(grades),
            "semester_date": random_date().strftime("%Y-%m-%d"),
            "credits": str(random.choice([3, 3, 3, 4, 4, 6])),
            "status": random.choice(["passed", "failed", "incomplete", "withdrawn"]),
        }
        clean_rows.append(row)

    messy_rows = []
    for row in clean_rows:
        messy = dict(row)
        if random.random() < 0.05:
            messy_rows.append(dict(row))

        messy["name"] = corrupt_name(messy["name"])
        messy["subject"] = random.choice(
            [
                messy["subject"].upper(),
                messy["subject"].lower(),
                f"  {messy['subject']}  ",
                messy["subject"],
            ]
        )
        messy["semester_date"] = corrupt_date(messy["semester_date"])
        messy["status"] = random.choice(
            [
                messy["status"].upper(),
                messy["status"].lower(),
                f"  {messy['status']}  ",
                messy["status"],
            ]
        )

        for field in ["credits", "status"]:
            if maybe_missing(0.08) is not None:
                messy[field] = ""

        messy_rows.append(messy)

    return clean_rows, messy_rows


def write_csv(filepath, rows, fieldnames):
    """Write rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for data cleaning agent"
    )
    parser.add_argument(
        "--output-dir", default="data_clean_env/tasks", help="Output directory"
    )
    parser.add_argument("--num-samples", type=int, default=30, help="Rows per dataset")
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=5,
        help="Number of dataset pairs to generate",
    )
    args = parser.parse_args()

    generators = [
        ("hr", generate_hr_data),
        ("healthcare", generate_healthcare_data),
        ("finance", generate_finance_data),
        ("logistics", generate_logistics_data),
        ("education", generate_education_data),
    ]

    print(
        f"Generating {args.num_datasets} dataset pairs with {args.num_samples} rows each..."
    )

    for i, (domain, gen_fn) in enumerate(generators[: args.num_datasets]):
        seed = 42 + i * 100
        clean_rows, messy_rows = gen_fn(n_rows=args.num_samples, seed=seed)
        fieldnames = list(clean_rows[0].keys())

        clean_path = os.path.join(args.output_dir, f"{domain}_clean.csv")
        messy_path = os.path.join(args.output_dir, f"{domain}_messy.csv")

        write_csv(clean_path, clean_rows, fieldnames)
        write_csv(messy_path, messy_rows, fieldnames)

        print(
            f"  ✅ {domain}: {len(messy_rows)} messy rows → {len(clean_rows)} clean rows"
        )
        print(f"     Messy: {messy_path}")
        print(f"     Clean: {clean_path}")

    print(f"\nDone! Generated {args.num_datasets} dataset pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
