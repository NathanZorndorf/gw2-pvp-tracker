import requests
import sys

def get_map_ids():
    # Target maps to find
    target_maps = {
        "Battle of Kyhlo",
        "Forest of Niflhel",
        "Legacy of the Foefire",
        "Temple of the Silent Storm",
        "Skyhammer",
        "Spirit Watch",
        "Courtyard",
        "Revenge of the Capricorn",
        "Eternal Coliseum",
        "Djinn's Dominion",
        "Divinity's Reach",
        "Lion's Arch",
        "The Grove",
        "Rata Sum",
        "Black Citadel",
        "Hoelbrak",
        "Heart of the Mists"
    }

    try:
        # Get all map IDs
        response = requests.get("https://api.guildwars2.com/v2/maps")
        response.raise_for_status()
        all_ids = response.json()
        
        # We'll fetch in chunks of 200 (API limit)
        chunk_size = 200
        found_maps = {}
        
        print(f"Total maps to search: {len(all_ids)}")
        
        for i in range(0, len(all_ids), chunk_size):
            chunk = all_ids[i:i + chunk_size]
            ids_str = ",".join(map(str, chunk))
            
            details_url = f"https://api.guildwars2.com/v2/maps?ids={ids_str}"
            details_response = requests.get(details_url)
            details_response.raise_for_status()
            
            maps_data = details_response.json()
            
            for m in maps_data:
                map_name = m.get("name")
                map_id = m.get("id")
                
                if map_name in target_maps:
                    found_maps[map_name] = map_id
                    
        # Verification
        print("MapID,MapName")
        
        # PvP Maps
        pvp_maps = [
            "Battle of Kyhlo", "Forest of Niflhel", "Legacy of the Foefire", 
            "Temple of the Silent Storm", "Skyhammer", "Spirit Watch", 
            "Courtyard", "Revenge of the Capricorn", "Eternal Coliseum", "Djinn's Dominion"
        ]
        
        for name in pvp_maps:
            if name in found_maps:
                print(f"{found_maps[name]},{name}")
            else:
                print(f"NOT FOUND,{name}")

        # Cities
        cities = [
            "Divinity's Reach", "Lion's Arch", "The Grove", 
            "Rata Sum", "Black Citadel", "Hoelbrak"
        ]
        
        for name in cities:
            if name in found_maps:
                print(f"{found_maps[name]},{name}")
            else:
                print(f"NOT FOUND,{name}")
                
        # Lobby
        if "Heart of the Mists" in found_maps:
            print(f"{found_maps['Heart of the Mists']},Heart of the Mists")
        else:
            print("NOT FOUND,Heart of the Mists")

        # Specific Verifications
        if "Divinity's Reach" in found_maps:
            print(f"\nVerification: Divinity's Reach ID is {found_maps['Divinity\'s Reach']} (Expected 18)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_map_ids()
