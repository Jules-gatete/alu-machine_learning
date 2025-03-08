#!/usr/bin/env python3
"""Script that provides some stats about Nginx logs stored in MongoDB"""

from pymongo import MongoClient

if __name__ == '__main__':
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['your_database']
        collection = db['nginx']

        # Insert logs (example)
        collection.insert_one({"method": "GET", "status_check": True})

        # Fetch logs
        logs = list(collection.find())

        # Process logs
        process_logs(logs)

    except Exception as e:
        print(f"Error: {e}")

def process_logs(logs):
    method_counts = {
        "GET": 0,
        "POST": 0,
        "PUT": 0,
        "PATCH": 0,
        "DELETE": 0
    }
    status_check_count = 0

    for log in logs:
        method = log.get("method")
        if method in method_counts:
            method_counts[method] += 1
        if log.get("status_check"):
            status_check_count += 1

    total_logs = len(logs)
    print(f"{total_logs} logs")
    print("Methods:")
    for method, count in method_counts.items():
        print(f"method {method}: {count}")
    print(f"{status_check_count} status check")

    print("Logs being processed:", logs)
