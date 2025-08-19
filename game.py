import random

def roll_dice():
    roll = random.randint(1, 6)
    return roll

def player_turn(player_pos, roll_history, pos_history, grid_size):
    dice_roll = roll_dice()
    roll_history.append(dice_roll)
    player_new_pos = None

    if player_pos + dice_roll <= grid_size:
        player_new_pos = player_pos + dice_roll
    else:
        player_new_pos = player_pos

    pos_history.append(player_new_pos)
    
    return player_new_pos
    

def game(player_pos, roll_history, pos_history, grid_size):
    turns = 1
    
    while (player_pos["1"] != grid_size) or (player_pos["2"] != grid_size) or (player_pos["3"] != grid_size) or (player_pos["4"] != grid_size):
        print(f"--------- Turn {turns} ---------------")
        player = 1
        
        while player <= 4:
            
            player_key = str(player)
            new_pos = player_turn(player_pos[player_key], roll_history[player_key], pos_history[player_key], grid_size)
            
            print(
                f"Player {player} rolls: {roll_history[player_key][-1]}; New Position: {pos_history[player_key][-1]}")
            
            if new_pos is not None:
                player_pos[player_key] = new_pos
                
            # print(player_key, player_pos[player_key])
            if player_pos[player_key] == grid_size:
                return player_key
            
            player += 1
        turns += 1
        
    print("\n\n")
        
def main():
    print("Snake and Ladder Game of 4 players:")
    size = int(input("Enter grid size: "))

    grid_size = size * size
    
    player_pos = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }

    roll_history = {
        "1": [],
        "2": [],
        "3": [],
        "4": []
    }

    pos_history = {
        "1": [],
        "2": [],
        "3": [],
        "4": []
    }
    
    winner = game(player_pos, roll_history, pos_history, grid_size)
    
    print(f"\nWinner: Player {winner}\n")
    
    print("------Data---------")
    
    i = 1
    while i <= 4:
        player_key = str(i)
        print(f"Player : {player_key}")
        print(f"Roll History: {roll_history[player_key]}")
        print(f"Position History: {pos_history[player_key]}")
        print(f"Status : {"WON" if winner==str(i) else "LOST"}\n")
        i += 1
    
if __name__ == "__main__":
    main()