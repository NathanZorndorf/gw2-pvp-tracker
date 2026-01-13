import requests

def check_map_types():
    ids_to_check = [
        549, 554, 795, 875, 900, 894, 1275, 1163, 1171, 1305, # PvP
        18, 50, 91, 139, 218, 326, # Cities
        350 # Lobby
    ]
    
    ids_str = ",".join(map(str, ids_to_check))
    url = f"https://api.guildwars2.com/v2/maps?ids={ids_str}"
    
    resp = requests.get(url)
    data = resp.json()
    
    print(f"{'ID':<6} {'Name':<30} {'Type':<15} {'Validation'}")
    print("-" * 70)
    
    for m in data:
        print(f"{m['id']:<6} {m['name']:<30} {m['type']:<15}")

if __name__ == "__main__":
    check_map_types()
