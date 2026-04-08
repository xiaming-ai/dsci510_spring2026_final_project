# fetch_acs_subject.py
import os
import sys
import json
import requests
import pandas as pd


def fetch_acs_subject_group(
    year: int = 2024,
    survey: str = "acs1",
    group: str = "S1901",
    ucgid: str = "0100000US",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a Census ACS subject table group and return a DataFrame.

    Example endpoint:
    https://api.census.gov/data/2024/acs/acs1/subject?get=group(S1901)&ucgid=0100000US
    """
    base_url = f"https://api.census.gov/data/{year}/acs/{survey}/subject"

    params = {
        "get": f"group({group})",
        "ucgid": ucgid,
    }

    # Optional Census API key
    if api_key:
        params["key"] = api_key

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unexpected API response format: {data}")

    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)

    return df


def main():
    # Read API key from environment if you have one; otherwise leave it blank.
    api_key = os.getenv("CENSUS_API_KEY")

    try:
        df = fetch_acs_subject_group(
            year=2024,
            survey="acs1",
            group="S1901",
            ucgid="0100000US",
            api_key=api_key,
        )
    except requests.HTTPError as e:
        print("HTTP error while calling Census API:", e)
        sys.exit(1)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

    print("Preview:")
    print(df.head())

    out_csv = "acs_s1901_us_2024.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Optional: save raw JSON too
    out_json = "acs_s1901_us_2024.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_json}")


if __name__ == "__main__":
    main()