#!/usr/bin/env python3
"""Pipeline Api"""
import requests


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)

    # Initialize an empty dictionary to count rockets
    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1

    # Define a specific order for rocket names
    rocket_order = ["5e9d0d95eda69955f709d1eb",  # Falcon 9
                    "5e9d0d95eda69974e44b38c5",  # Falcon Heavy
                    "5e9d0d95eda69955f709d1ec"]  # Falcon 1

    # Sort and print the rocket names and their counts
    for key in rocket_order:
        if key in rocket_dict:
            rurl = "https://api.spacexdata.com/v4/rockets/" + key
            req = requests.get(rurl)
            print(req.json()["name"] + ": " + str(rocket_dict[key]))
