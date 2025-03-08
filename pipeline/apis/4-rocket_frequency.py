#!/usr/bin/env python3
"""Pipeline API - Fetches SpaceX launch data and counts rocket usage."""
import requests
import sys

if __name__ == '__main__':
    """Fetches launch and rocket data from SpaceX API"""
    url = "https://api.spacexdata.com/v4/launches"

    try:
        r = requests.get(url, timeout=10)  # Add timeout for reliability
        r.raise_for_status()  # Raises error for HTTP status codes >= 400
    except requests.exceptions.RequestException as e:
        print("Error fetching launch data: {}".format(e))
        sys.exit(1)

    # Initialize an empty dictionary to count rockets
    rocket_dict = {}

    try:
        launches = r.json()
        for launch in launches:
            rocket_id = launch.get("rocket")
            if rocket_id:
                rocket_dict[rocket_id] = rocket_dict.get(rocket_id, 0) + 1
    except ValueError:
        print("Invalid JSON response for launch data")
        sys.exit(1)

    # Fetch all rocket names dynamically
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    try:
        rockets_response = requests.get(rockets_url, timeout=10)
        rockets_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error fetching rocket data: {}".format(e))
        sys.exit(1)

    try:
        rocket_names = {
            r["id"]: r["name"] for r in rockets_response.json()
        }
    except ValueError:
        print("Invalid JSON response for rocket data")
        sys.exit(1)

    # Sort and print the rocket names and their counts
    for rocket_id, count in sorted(
        rocket_dict.items(), key=lambda x: x[1], reverse=True
    ):
        rocket_name = rocket_names.get(rocket_id, "Unknown Rocket")
        print("{}: {}".format(rocket_name, count))
