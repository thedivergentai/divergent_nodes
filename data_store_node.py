import os

NODE_MEMORY_STACKS = {} # Class-level dictionary to store memory stacks

class DataStoreNode:
    NODE_DISPLAY_NAME = "Data Store"
    RETURN_TYPES = ()
    CATEGORY = "Divergent Nodes 👽/Data Storage"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*",),
                "directory": ("STRING", {"default": "store"}),
                "memory_stack_size": ("INT", {"default": 1, "min": 1}),
            }
        }

    def __init__(self):
        self.stack = [] # Instance-level stack (not used for persistent memory across executions)


    def store_data(self, data, directory, memory_stack_size):
        stack_size = int(memory_stack_size)
        if stack_size < 1:
            stack_size = 1

        default_directory = "store" # Default to "store" in the repo root
        output_dir = directory or default_directory
        os.makedirs(output_dir, exist_ok=True)

        node_id = id(self) # Get unique ID for the node instance
        if node_id not in NODE_MEMORY_STACKS:
            NODE_MEMORY_STACKS[node_id] = []
        current_stack = NODE_MEMORY_STACKS[node_id]


        current_stack.append(data)
        if len(current_stack) > stack_size:
            current_stack.pop(0) # FIFO

        NODE_MEMORY_STACKS[node_id] = current_stack # Update the class-level dictionary

        for i, item in enumerate(current_stack):
            file_path = os.path.join(output_dir, f"stack_{i+1}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(repr(item)) # Store data as string


        return () # No output for now
