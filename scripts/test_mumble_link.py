import sys
import os
import time

# Add the project root to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integration.mumble_link import MumbleLink, PVP_MAPS

def main():
    print("Initializing MumbleLink reader...")
    # Try to find the CSV file relative to this script
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'map_ids.csv')
    csv_path = os.path.abspath(csv_path)
    
    if os.path.exists(csv_path):
        print(f"Loading map names from {csv_path}")
        ml = MumbleLink(map_csv_path=csv_path)
    else:
        print("Map CSV not found, using defaults.")
        ml = MumbleLink()

    if not ml.is_active:
        print("Could not connect to MumbleLink. Make sure Guild Wars 2 is running.")
        return

    print("Connected! Reading data (Ctrl+C to stop)...")
    print("-" * 50)
    
    try:
        while True:
            ml.read()
            
            # Debugging raw values even if 0
            ui_version = ml.data.uiVersion if ml.data else -1
            ui_tick = ml.data.uiTick if ml.data else -1
            
            if ml.data and ml.data.uiTick > 0:
                identity = ml.get_identity()
                player_name = identity.get("name", "Unknown")
                map_id = identity.get("map_id", 0)
                map_name = ml.get_map_name()
                
                # Context data
                ctx = ml.context_obj
                player_pos = f"X: {ctx.playerX:.2f}, Y: {ctx.playerY:.2f}" if ctx else "N/A"
                
                print(f"Tick: {ui_tick} | Ver: {ui_version} | Player: {player_name} | Map: {map_name} | Pos: {player_pos}     ", end='\r')
            else:
                print(f"Waiting for game data... (uiTick: {ui_tick}, uiVersion: {ui_version})", end='\r')
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ml.close()

if __name__ == "__main__":
    main()
