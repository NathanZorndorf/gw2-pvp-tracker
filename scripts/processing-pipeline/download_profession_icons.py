import requests
import os
from urllib.parse import urlparse, unquote

def download_profession_icons(output_dir="data/profession_icons"):
    """
    Download profession icon PNG files with 'icon_white' in the name from Guild Wars 2 wiki.
    """
    os.makedirs(output_dir, exist_ok=True)

    # MediaWiki API endpoint
    api_url = "https://wiki.guildwars2.com/api.php"

    # Get all file members of the category
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Profession_icons",
        "cmtype": "file",
        "cmlimit": 500,
        "format": "json"
    }

    response = requests.get(api_url, params=params)
    data = response.json()

    if "query" not in data or "categorymembers" not in data["query"]:
        print("Failed to fetch category members")
        return

    members = data["query"]["categorymembers"]

    # Filter for files with 'icon_white' in the title
    icon_white_files = [member for member in members if "icon white" in member["title"].lower()]

    print(f"Found {len(icon_white_files)} icon_white files")

    for file_info in icon_white_files:
        title = file_info["title"]
        # Get image info
        imageinfo_params = {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }

        img_response = requests.get(api_url, params=imageinfo_params)
        img_data = img_response.json()

        pages = img_data["query"]["pages"]
        page_id = list(pages.keys())[0]
        if "imageinfo" in pages[page_id]:
            image_url = pages[page_id]["imageinfo"][0]["url"]
            # Extract filename
            parsed_url = urlparse(image_url)
            filename = unquote(os.path.basename(parsed_url.path))

            # Modify filename to just the first word
            base_name = filename.split('_')[0]
            new_filename = f"{base_name}.png"

            # Download the image
            img_download = requests.get(image_url)
            if img_download.status_code == 200:
                filepath = os.path.join(output_dir, new_filename)
                with open(filepath, 'wb') as f:
                    f.write(img_download.content)
                print(f"Downloaded: {new_filename}")
            else:
                print(f"Failed to download: {new_filename}")
        else:
            print(f"No image info for: {title}")

if __name__ == "__main__":
    download_profession_icons()