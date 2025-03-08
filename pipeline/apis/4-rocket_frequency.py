#!/usr/bin/env python3
"""Pipeline Api"""
import requests

if __name__ == '__main__':
    """Pipeline API"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)

    if r.status_code != 200:
        print("Failed to fetch launch data")
        exit(1)

    # Initialize an empty dictionary to count rockets
    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch["rocket"]
        rocket_dict[rocket_id] = rocket_dict.get(rocket_id, 0) + 1

    # Fetch all rocket names dynamically
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets_response = requests.get(rockets_url)

    if rockets_response.status_code != 200:
        print("Failed to fetch rocket data")
        exit(1)

    rocket_names = {r["id"]: r["name"] for r in rockets_response.json()}

    # Sort and print the rocket names and their counts
    for rocket_id, count in sorted(rocket_dict.items(), key=lambda x: x[1], reverse=True):
        if rocket_id in rocket_names:
            print(f"{rocket_names[rocket_id]}: {count}")
