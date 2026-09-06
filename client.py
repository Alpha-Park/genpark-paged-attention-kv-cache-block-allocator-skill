import math
from typing import Dict, Any, List, Optional

class PagedAttentionKVCacheBlockAllocator:
    """
    Manages non-contiguous physical memory blocks for generative LLM sequences,
    mimicking the OS virtual memory paging to eliminate KV-cache internal fragmentation.
    """
    def __init__(self, block_size_tokens: int = 16, total_physical_blocks: int = 64):
        self.block_size_tokens = block_size_tokens
        self.total_physical_blocks = total_physical_blocks
        self.free_blocks = list(range(total_physical_blocks))
        self.block_tables: Dict[str, List[int]] = {}

    def allocate_sequence(self, seq_id: str, prompt_token_count: int) -> Dict[str, Any]:
        needed_blocks = math.ceil(prompt_token_count / self.block_size_tokens)
        if len(self.free_blocks) < needed_blocks:
            return {"status": "OUT_OF_MEMORY", "seq_id": seq_id, "free_blocks": len(self.free_blocks)}

        allocated = []
        for _ in range(needed_blocks):
            allocated.append(self.free_blocks.pop(0))

        self.block_tables[seq_id] = allocated

        return {
            "status": "ALLOCATED",
            "seq_id": seq_id,
            "token_count": prompt_token_count,
            "blocks_allocated": len(allocated),
            "physical_block_ids": allocated,
            "remaining_free_blocks": len(self.free_blocks)
        }

    def append_token(self, seq_id: str, current_token_count: int) -> Dict[str, Any]:
        if seq_id not in self.block_tables:
            return {"status": "NOT_FOUND"}

        table = self.block_tables[seq_id]
        current_capacity = len(table) * self.block_size_tokens

        # Check if new block needed
        if current_token_count >= current_capacity:
            if not self.free_blocks:
                return {"status": "OUT_OF_MEMORY"}
            new_block = self.free_blocks.pop(0)
            table.append(new_block)
            return {"status": "EXPANDED", "new_physical_block": new_block, "total_blocks": len(table)}

        return {"status": "OK", "slot_available": current_capacity - current_token_count}

    def free_sequence(self, seq_id: str) -> Dict[str, Any]:
        if seq_id in self.block_tables:
            freed = self.block_tables.pop(seq_id)
            self.free_blocks.extend(freed)
            return {"status": "FREED", "seq_id": seq_id, "freed_count": len(freed), "free_blocks": len(self.free_blocks)}
        return {"status": "NOT_FOUND"}
