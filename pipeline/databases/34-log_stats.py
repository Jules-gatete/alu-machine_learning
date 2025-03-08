#!/usr/bin/env python3
"""Script that provides some stats about Nginx logs stored in MongoDB"""

from pymongo import MongoClient

if __name__ == '__main__':
    try:
        client = MongoClient('mongodb://127.0.0.1:27017')
        collection = client.logs.nginx

        # Total number of documents
        total_logs = collection.count_documents({})

        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        method_counts = {method: collection.count_documents({"method": method}) for method in methods}

        # Count of status check
        status_check_count = collection.count_documents({"method": "GET", "path": "/status"})

        # Print output
        print(f"{total_logs} logs")
        print("Methods:")
        for method in methods:
            print(f"\tmethod {method}: {method_counts[method]}")
        print(f"{status_check_count} status check")

    except Exception as e:
        print(f"Error: {e}")
