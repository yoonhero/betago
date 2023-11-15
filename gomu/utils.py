import json
import time

GAME_INFO_KEY = "info"
RECORD_KEY = "record"

# return ms timestamp 
def get_timestamp() -> int:
    return int(time.time() * 1000)

def save_record(current_episode, game_info, record, to_json=False) -> None:
    if to_json:
        with open(f"./tmp/record_{current_episode}_{str(get_timestamp())}.json", "a", encoding="utf-8") as f:
            to_save_data = {"info": game_info, "record": record}
            json.dump(to_save_data, f)
    else:
        with open(f"./tmp/record_{current_episode}_{str(get_timestamp())}.txt", "w", encoding="utf-8") as f:
            # nrow, ncol, ntowin
            nrow, ncol, ntowin = game_info["nrow"], game_info["ncol"], game_info["n_to_win"]
            to_save_data = f"{nrow},{ncol},{ntowin}\n" 
            to_save_data += "\n".join(record)
            f.write(to_save_data)

def open_record(record_path, from_json=False) -> object:
    with open(record_path, "r", encoding="utf-8") as f:
        if from_json:
            record = json.open(f)
            return record
        else:
            data = f.read()
            tmp = data.split("\n")
            header = tmp[0]
            body = tmp[1:]
            nrow, ncol, ntowin = [int(param) for param in header.split(",")]
            game_info = {"nrow": nrow, "ncol": ncol, "n_to_win": ntowin}
            record = [[int(i) for i in pos.strip().split(",")] for pos in body]
            
            return {"game_info": game_info, "record": record}