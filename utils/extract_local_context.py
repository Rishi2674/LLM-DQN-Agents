import numpy as np

# Define the maze where 0 is the road and 1 is the wall
# maze = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],      
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
#     [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
#     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
#     [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],   
#     [1, 1, 0, 0, 0, 1, 0, 1, 0 ,1],
#     [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1],
# ])

def extract_local_context(maze: np.array , agent_position: tuple , size: int = 3) -> np.array:
    """
    Extracts a local context (sub-grid) around the agent's position in the maze.

    Args:
        maze (np.array): The maze represented as a binary array (road = '0', wall = '1').
        agent_position (tuple): The (row,col) coordinates of the agent.
        size (int): The size of the local context (must be an odd number).

    Returns:
        np.array: A binary array representing the local context around the agent.
    """
    agent_row , agent_col = agent_position
    half_size = size //2

    # Calculate boundaries for local view
    start_row = max(0 , agent_row - half_size)
    end_row = min(maze.shape[0] , agent_row + half_size +1)
    start_col = max(0 , agent_col - half_size)
    end_col = min(maze.shape[1] , agent_col + half_size +1)

    # Extract local view
    local_view = maze[start_row:end_row , start_col:end_col]

    # Pad local view if it's smaller than desired size
    pad_top = half_size - (agent_row - start_row)
    pad_bottom = half_size - (end_row - agent_row -1)
    pad_left = half_size - (agent_col - start_col)
    pad_right = half_size - (end_col - agent_col -1)

    local_context = np.pad(
        local_view,
        ((pad_top , pad_bottom) , (pad_left , pad_right)),
        mode='constant',
        constant_values=1 # Padding with walls (value of 'wall' which is '1')
    )

    return local_context

def represent_local_context(local_context: np.array) -> str:
    """
    Converts the local context (sub-grid) to a string representation.

    Args:
        local_context (np.array): A binary array representing the local context.

    Returns:
        str: A string representation of the local context.
    """
    rows = [''.join(str(cell) for cell in row) for row in local_context]
    return '\n'.join(rows)

# Example usage:
# agent_position = (5 ,5) # Example agent position
# local_context = extract_local_context(maze , agent_position)
# local_context_str = represent_local_context(local_context)

# print("Local Context (3x3):\n",local_context_str)

# # Example usage near corner
# agent_position_corner = (2 ,2) # Example agent position near a corner
# local_context_corner = extract_local_context(maze , agent_position_corner)
# local_context_str_corner = represent_local_context(local_context_corner)

# print("\nLocal Context (3x3) near corner:\n", local_context_str_corner)
